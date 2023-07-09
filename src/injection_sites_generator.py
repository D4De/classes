from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import numpy as np
import struct
from src.pattern_generators import generator_functions

from src.utils import random_choice, unpack_table
from src.loggers import get_logger
from src.visualize import visualize

logger = get_logger('InjectionSiteGenerator')

class InjectableSite(object):
    """
    Describes an injectable operator and it is characterized by the TensorFlow's operator type,
    the TensorFlow's operator graph name and the size of the output tensor.
    """

    def __init__(self, operator_name : str, size : str):
        """
        Creates the object with the operator type, name and size.

        Arguments:
            operator_type {OperatorType} -- TensorFlow's operator type.
            operator_name {str} -- TensorFlow's operator graph name.
            size {str} -- The output tensor size expressed as string.
        """
        self.__operator_name = operator_name
        size = size.replace("None", "1")
        tuple_size = eval(size)

        # If the size has less than 4 components, it is expanded to match a tensor shape.
        if len(tuple_size) == 4:
            self.__has_all_components = True
            self.__size = size
            self.__components = 0
        else:
            remainder = 4 - len(tuple_size)
            self.__has_all_components = False
            self.__components = remainder
            self.__size = str(tuple([1] * remainder + list(tuple_size)))

    def __repr__(self):
        return "InjectableSite[Type: {}, Name: {}, Size: {}]".format(self.__operator_name,
                                                                     self.__size)

    def __str__(self):
        return self.__repr__()

    def __get__size(self):
        if self.__has_all_components:
            return self.__size
        else:
            size_eval = eval(self.__size)
            size = [size_eval[i] for i in range(self.__components, len(size_eval))]
            return str(tuple(size))

    operator_type = property(fget=lambda self: self.__operator_type)
    operator_name = property(fget=lambda self: self.__operator_name)
    size = property(fget=__get__size)


class InjectionValue(object):
    """
    Represents a value to be injected in an operator's output tensor.
    There can be 4 types of values:
    NaN, inserts a NaN value, Zeroes, inserts a zero value,
    [-1, 1], inserts a difference between -1 and 1 (zero excluded) and
    Others, which represents a random 32-wide bitstring.
    """

    NAN = 1
    ZERO = 2
    IN_RANGE = 3
    OUT_OF_RANGE = 4

    @staticmethod
    def nan():
        return InjectionValue(InjectionValue.NAN, np.float32(np.nan))

    @staticmethod
    def zeroes():
        return InjectionValue(InjectionValue.ZERO, np.float32(0.0))

    @staticmethod
    def in_range():
        raw_value = np.random.uniform(0.0, 1.0)
        return InjectionValue(InjectionValue.IN_RANGE, np.float32(raw_value))

    @staticmethod
    def out_of_range():
        bitstring = "".join(np.random.choice(["0", "1"], size=32))
        integer_bitstring = int(bitstring, base=2)
        float_bitstring = np.frombuffer(np.array(integer_bitstring), dtype=np.float32)[0]
        return InjectionValue(InjectionValue.OUT_OF_RANGE, np.float32(float_bitstring))
    


    def __init__(self, value_type, raw_value):
        self.__value_type : Literal[1,2,3,4] = value_type
        self.__raw_value : np.float32 = raw_value

    def __str__(self):
        return "({}, {})".format(self.__value_type, hex(struct.unpack('<I', struct.pack('<f', self.__raw_value))[0]))
    
    def __repr__(self):
        return f"(Inj. Value type: {self.__value_type} Value: {self.__raw_value})"

    def get_value(self, range_min, range_max):
        if self.__value_type == InjectionValue.ZERO:
            return 0.0
        elif self.__value_type == InjectionValue.NAN:
            return float('nan')
        elif self.__value_type == InjectionValue.IN_RANGE:
            return range_min + self.__raw_value * (range_max - range_min)
        else:
            if range_min <= self.__raw_value <= range_max:
                return self.__raw_value + (range_max - range_min) * np.random.choice([-1, 1]) * np.random.uniform(2.0, 600.0)
            else:
                return self.__raw_value

    @property
    def value_type(self):
        return self.__value_type
    
    @property
    def raw_value(self):
        return self.__raw_value



class InjectionSite(object):
    """
    Represents an injection site and is composed by the operator name to inject,
    the indexes where insert the injections and the values to insert.

    It can be iterated to get pairs of indexes and values.
    """

    def __init__(self, operator_name, output_shape):
        self.__operator_name = operator_name
        self.__indexes : List[List[int]] = []
        self.__values : List[InjectionValue] = []
        self.__output_shape = output_shape

    def add_injection(self, index, value):
        self.__indexes.append(index)
        self.__values.append(value)

    def __iter__(self):
        self.__iterator = zip(self.__indexes, self.__values)
        return self

    def __len__(self) -> int:
        return len(self.__indexes)

    def __repr__(self):
        return f'{self.__operator_name} {self.__indexes} {self.__values}'

    def next(self):
        next_element = next(self.__iterator)
        if next_element is None:
            raise StopIteration
        else:
            return next_element

    def get_indexes_values(self) -> Iterable[Tuple[List[int], InjectionValue]]:
        return zip(self.__indexes, self.__values)

    def as_indexes_list(self) -> List[int]:
        return list(self.__indexes)

    operator_name = property(fget=lambda self: self.__operator_name)

    def to_json(self):
        json_representation = {}
        for index, value in self:
            json_representation[str(index)] = str(value)
        json_representation["operator_name"] = self.__operator_name
        return json_representation
    
    def visualize(self, output_path : str):
        tensor_diff = np.zeros(self.__output_shape)
        faulty_channels = set()
        for i in range(len(self)):
            n, c, h, w = self.__indexes[i]
            faulty_channels.add(c)
            tensor_diff[n, c, h, w] = self.__values[i].value_type
        visualize(tensor_diff, faulty_channels, 'NCHW', output_path, save=True, invalidate=True)

MAX_FAILED_ATTEMPTS_IN_A_ROW = 20


class InjectionSitesGenerator(object):
    def __init__(self, injectable_sites, models_folder: str = '/models', fixed_spatial_class : Optional[str] = None, fixed_domain_class : Optional[dict] = None):
        self.__injectable_sites = injectable_sites
        self.__models = self.__load_models(models_folder)
        self.__fixed_spatial_class = fixed_spatial_class
        self.__fixed_domain_class = fixed_domain_class

    def generate_random_injection_sites(self, size: int, fixed_spatial_class : Optional[str] = None, fixed_domain_class : Optional[dict] = None) -> List[InjectionSite]:

        injectables_site_indexes = np.random.choice(len(self.__injectable_sites), size=size)
        injection_sites = []

        for index in injectables_site_indexes:
            extracted = False
            attempts = 0
            while not extracted and attempts < MAX_FAILED_ATTEMPTS_IN_A_ROW: 
                try:
                    injectable_site = self.__injectable_sites[index]
                    operator_name = injectable_site.operator_name  
                    output_shape = eval(injectable_site.size)
                    injection_site = InjectionSite(operator_name, output_shape)
                    if fixed_spatial_class is not None:
                        spatial_class = fixed_spatial_class
                    elif self.__fixed_spatial_class is not None:
                        spatial_class = self.__fixed_spatial_class
                    else:
                        spatial_class = self.__select_spatial_class(operator_name)
                    if fixed_domain_class is not None:
                        spatial_class = fixed_domain_class
                    if self.__fixed_domain_class is not None:
                        domain_class = self.__fixed_domain_class
                    else:
                        domain_class = self.__select_domain_class(operator_name, spatial_class)
                    spatial_parameters = self.__select_spatial_parameters(operator_name, spatial_class)
                    spatial_positions = self.__generate_spatial_pattern(spatial_class, output_shape, spatial_parameters)
                    if spatial_positions is None:
                        logger.error(f"Injection attempt #{attempts + 1} failed. Params {spatial_class=} {spatial_parameters=} {output_shape=}")
                        raise RuntimeError("Injection attempt failed")

                    channel_count = len(set(sp_pos[1] for sp_pos in spatial_positions))
                    logger.info(f"Injection details. Spatial: {spatial_class} {spatial_parameters} Domain: {domain_class}. Cardinality: {len(spatial_positions)} Channel Count: {channel_count}")
                    logger.debug(spatial_positions)
                    corrupted_values = self.__generate_domains(domain_class, len(spatial_positions))
                    for idx, value in zip(spatial_positions, corrupted_values):
                        injection_site.add_injection(idx, value)                
                    injection_sites.append(injection_site)
                    extracted = True
                except Exception as e:
                    logger.error(f"Failed to inject. Attempts in a row: {attempts + 1}")
                    logger.error(e, exc_info=True)
                finally:
                    attempts += 1
            if attempts == MAX_FAILED_ATTEMPTS_IN_A_ROW:
                logger.error(f"Injection failed {MAX_FAILED_ATTEMPTS_IN_A_ROW}. Aborting")
                raise RuntimeError(f"Failed to inject {MAX_FAILED_ATTEMPTS_IN_A_ROW} in a row")
        return injection_sites
    
    def __select_spatial_class(self, operator_name : str) -> str:
        if operator_name not in self.__models:
            raise KeyError(f"Operator {operator_name} does not exists. Check if a file name {operator_name}.json is included in the selected model folder.")
        sp_classes, sp_class_description = unpack_table(self.__models[operator_name])
        sp_classes_freqs = [desc["frequency"] for desc in sp_class_description]
        return random_choice(sp_classes, p=sp_classes_freqs)
    
    def __select_domain_class(self, operator_name : str, spatial_class : str) -> Dict[str, Any]:
        dom_classes : List[Dict[str, Any]] = self.__models[operator_name][spatial_class]["domain_classes"]
        return random_choice(dom_classes, p=[dc["frequency"] for dc in dom_classes])

    def __select_spatial_parameters(self, operator_name : str, spatial_class : str) -> Dict[str, Any]:
        sp_parameters : List[Dict[str, Any]] = self.__models[operator_name][spatial_class]["parameters"]
        selected_params = random_choice(sp_parameters, p=[dc["conditional_frequency"] for dc in sp_parameters])        
        return {**selected_params["keys"], **selected_params["stats"]}


    def __generate_spatial_pattern(self, spatial_class : str, output_shape : List[int], parameters : Dict[str, Any]) -> Optional[List[Iterable[int]]]:
        is_random = any(isinstance(val, dict) and "RANDOM" in val for val in parameters.values())
        if is_random:
            parameters = self.__realize_random_parameters(parameters)
        if spatial_class in generator_functions:
            raveled_positions = generator_functions[spatial_class](output_shape, parameters)
            if raveled_positions is None:
                return None 
            return [np.unravel_index(rp, shape=output_shape) for rp in raveled_positions]
        else:
            raise NotImplementedError(f"{spatial_class} generator is not implemented. Check if it was included in src/pattern_generators/__init__.py dictionary")
        
    def __generate_domains(self, domain_class : str, cardinality : int):
        def inj_value_from_name(name : str):
            if name == "flip" or name == "out_of_range":
                return InjectionValue.out_of_range()
            if name == "zero":
                return InjectionValue.zeroes()
            if name == "in_range":
                return InjectionValue.in_range()
            if name == "nan":
                return InjectionValue.nan()
            else:
                raise ValueError(f"{name} class does not exists")
            
        
        if "random" in domain_class:
            val_class, freq = unpack_table(domain_class["values"])

            random_val_classes = random_choice(len(val_class), size=cardinality, replace=True, p=freq)


            return [inj_value_from_name(val_class[vc_id]) for vc_id in random_val_classes]

        value_frequencies = dict(domain_class)
        if "count" in value_frequencies:
            del value_frequencies["count"]
        if "frequency" in value_frequencies:    
            del value_frequencies["frequency"]
        value_classes, freq_ranges = unpack_table(value_frequencies)
        if len(value_classes) == 1:
          return [inj_value_from_name(value_classes[0]) for i in range(cardinality)]
        elif len(value_classes) == 2:
            rand_freq_1 = np.random.uniform(*freq_ranges[0])
            rand_freq_2 = 100 - rand_freq_1
            random_val_classes = random_choice(2, size=cardinality, replace=True, p=[rand_freq_1, rand_freq_2])
            return [inj_value_from_name(value_classes[vc_id]) for vc_id in random_val_classes]
        else:
            raise NotImplementedError("More than two value classes are not yet supported. Please make them fallback to random or implement code to support them")


    def __realize_random_parameters(self, parameters : Dict[str, Any]) -> Dict[str, Any]:
        realized_params : Dict[str, Any] = {}

        # There could be some keys that specify a minumum and a maximum constraint for a parameters (example min_span_width, max_span_width)
        # In this case the values that we randomly extract from a minium or a maximum value must be coherent (the max should be >= than the min)
        # In the models these minimum and maximum constraint can be detected by finding two keys that start min_<X> max_<X> where <X> is a string
        # in common
        min_keys = set(k[4:] for k in parameters.keys() if k.startswith("min_"))
        max_keys = set(k[4:] for k in parameters.keys() if k.startswith("max_"))
        min_max_constrained_parameters = min_keys & max_keys

        # First we insert all the non random keys
        for param_name, param_values in parameters.items():
            if isinstance(param_values, dict) and "RANDOM" in param_values:
                continue
            realized_params[param_name] = param_values
        # then we extract the random ones
        for param_name, param_values in parameters.items():
            if isinstance(param_values, dict) and "RANDOM" in param_values:
                # The param contains random values to be extracted
                random_values = param_values["RANDOM"]
                base_param_name = param_name[4:]
                # Check if the random parameter is constrained by a minimum and a maximum
                if base_param_name in min_max_constrained_parameters:
                    if param_name.startswith("min_"):
                        dual_parameter = f"max_{base_param_name}"
                        constraint = "max"
                    else:
                        dual_parameter = f"min_{base_param_name}"
                        constraint = "min"
                    # If the dual parameters were already generated or inserted in the realized_params dict we have to follow the constaint
                    if dual_parameter in realized_params:
                        dual_parameter_value = realized_params[dual_parameter]
                        # Filter out the random values that not follow the constraints
                        if constraint == "min":
                            eligible_values = [rv for rv in random_values if rv >= dual_parameter_value]
                        else:
                            eligible_values = [rv for rv in random_values if rv <= dual_parameter_value]
                        # For avoiding errors we add the constraint if no values were eligible
                        if len(eligible_values) == 0:
                            eligible_values = [dual_parameter_value]
                    else:
                        # We don't have to worry about the constrain, because the dual parameter is not yet realized
                        eligible_values = random_values
                else:
                    eligible_values = random_values
                # eligible_values contains the values that respect the constraints if any
                chosen_index = random_choice(len(eligible_values))
                realized_params[param_name] = eligible_values[chosen_index]

        return realized_params

    
    def __load_models(self, models_folder):
        spatial_models : Dict[str, Any] = {}
        available_operators = [file[:-5] for file in os.listdir(models_folder) if file.endswith(".json")]

        for model_operator_name in available_operators:
            
            spatial_model_path = os.path.join(models_folder,
                                              f'{model_operator_name}.json')
            with open(spatial_model_path, "r") as spatial_model_json:
                spatial_model = json.load(spatial_model_json)
                spatial_models[model_operator_name] = {k : v for k, v in spatial_model.items() if not k.startswith("_")}

        return spatial_models



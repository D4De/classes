from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import traceback
import struct
from pattern_generators import generator_functions

from utils import random_choice, unpack_table
from .operators import OperatorType


from .operators import OperatorType

import logging
import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)

# create logger
logger = logging.getLogger('InjectionSiteGenerator')


class InjectableSite(object):
    """
    Describes an injectable operator and it is characterized by the TensorFlow's operator type,
    the TensorFlow's operator graph name and the size of the output tensor.
    """

    def __init__(self, operator_type : OperatorType, operator_name : str, size : str):
        """
        Creates the object with the operator type, name and size.

        Arguments:
            operator_type {OperatorType} -- TensorFlow's operator type.
            operator_name {str} -- TensorFlow's operator graph name.
            size {str} -- The output tensor size expressed as string.
        """
        self.__operator_type = operator_type
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
        return "InjectableSite[Type: {}, Name: {}, Size: {}]".format(self.__operator_type, self.__operator_name,
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
        self.__value_type = value_type
        self.__raw_value = raw_value

    def __str__(self):
        return "({}, {})".format(self.__value_type, hex(struct.unpack('<I', struct.pack('<f', self.__raw_value))[0]))

    def get_value(self, range_min, range_max):
        if self.__value_type == InjectionValue.ZERO:
            return np.float32(0.0)
        elif self.__value_type == InjectionValue.NAN:
            return np.float32(np.nan)
        elif self.__value_type == InjectionValue.IN_RANGE:
            return np.float32(range_min + self.__raw_value * (range_max - range_min))
        else:
            if range_min <= self.__raw_value <= range_max:
                return np.float32(self.__raw_value + (range_max - range_min) * np.random.choice([-1, 1]) * np.random.uniform(2.0, 600.0))
            else:
                return np.float32(self.__raw_value)

    value_type = property(fget=lambda self: self.__value_type)
    raw_value = property(fget=lambda self: self.__raw_value)


class InjectionSite(object):
    """
    Represents an injection site and is composed by the operator name to inject,
    the indexes where insert the injections and the values to insert.

    It can be iterated to get pairs of indexes and values.
    """

    def __init__(self, operator_name):
        self.__operator_name = operator_name
        self.__indexes = []
        self.__values = []

    def add_injection(self, index, value):
        self.__indexes.append(index)
        self.__values.append(value)

    def __iter__(self):
        self.__iterator = zip(self.__indexes, self.__values)
        return self

    def next(self):
        next_element = next(self.__iterator)
        if next_element is None:
            raise StopIteration
        else:
            return next_element

    def get_indexes_values(self) -> Iterable[Tuple[int, int]]:
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


operator_names_table = {
    "S1_add": "AddV2",
    "S2_add": "Add",
    "S1_batch_norm": "FusedBatchNormV3",
    "S1_biasadd": "BiasAdd",
    "S1_convolution": "Conv2D1x1",
    "S1_div": "RealDiv",
    "S1_exp": "Exp",
    "S1_leaky_relu": "LeakyRelu",
    "S1_mul": "Mul",
    "S1_sigmoid": "Sigmoid",
    "S2_convolution": "Conv2D3x3",
    "S3_convolution": "Conv2D3x3S2",
    "S1_convolution_test": "Conv2D1x1"
}


MAX_FAILED_ATTEMPTS_IN_A_ROW = 20


class InjectionSitesGenerator(object):
    def __init__(self, injectable_sites, mode, models_folder: str = '/models'):
        self.__injectable_sites = injectable_sites
        self.__models = self.__load_models(models_folder)

    def generate_random_injection_sites(self, size: int) -> List[InjectionSite]:

        injectables_site_indexes = np.random.choice(len(self.__injectable_sites), size=size)
        injection_sites = []

        for index in injectables_site_indexes:
            injectable_site = self.__injectable_sites[index]
            operator_type = injectable_site.operator_type.name
            logger.info(f"Trying injection in operator: {operator_type}")
            injection_site = InjectionSite(injectable_site.operator_name)
            spatial_class = self.__select_spatial_class()
            domain_class = self.__select_domain_class(spatial_class)
            spatial_parameters = self.__select_spatial_parameters(spatial_class)

            for parameter_attempt in range(MAX_FAILED_ATTEMPTS_IN_A_ROW):
                spatial_positions = self.__generate_spatial_pattern(spatial_class, eval(injectable_site.size), spatial_parameters)
                if spatial_positions is not None:
                    break
                logger.warn(f"Injection attempt #{parameter_attempt + 1} failed using {spatial_parameters}, retrying again with the same parameters")
            else:
                logger.error(f"Injection failed for {MAX_FAILED_ATTEMPTS_IN_A_ROW} attempts while using {spatial_parameters}, retrying with other parameters")


            corrupted_values = self.__generate_domains(domain_class, len(spatial_positions))

            for idx, value in zip(spatial_positions, corrupted_values):
                injection_site.add_injection(idx, value)                
            injection_sites.append(injection_site)

        return injection_sites
    
    def __select_spatial_class(self) -> str:
        sp_classes, sp_class_description = unpack_table(self.__models)
        sp_classes_freqs = [desc["frequency"] for desc in sp_class_description]
        return random_choice(sp_classes, size=1, p=sp_classes_freqs)
    
    def __select_domain_class(self, spatial_class : str) -> Dict[str, Any]:
        dom_classes : List[Dict[str, Any]] = self.__models[spatial_class]["domain_classes"]
        return random_choice(dom_classes, size=1, p=[dc["frequency"] for dc in dom_classes])

    def __select_spatial_parameters(self, spatial_class : str) -> Dict[str, Any]:
        sp_parameters : List[Dict[str, Any]] = self.__models[spatial_class]["parameters"]
        selected_params = random_choice(sp_parameters, size=1, p=[dc["conditional_frequency"] for dc in sp_parameters])        
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
            raise NotImplementedError(f"{spatial_class} generator is not implemented")
        
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
        del value_frequencies["count"]
        del value_frequencies["frequency"]
        value_classes, freq_ranges = unpack_table(value_frequencies)
        selected_frequencies = []
        remaining = cardinality
        for i in range(len(value_classes) - 1):
            min_freq, max_freq = freq_ranges[i]
            rand_freq = np.random.uniform(min_freq, max_freq)
            remaining -= rand_freq
            selected_frequencies.append(rand_freq)
        if remaining < 0:
            raise ValueError("Domain classes probabilties sum more than 1.0")
        selected_frequencies.append(remaining)
        random_val_classes = random_choice(len(value_classes), size=cardinality, replace=True, p=selected_frequencies)

        return [inj_value_from_name(val_class[vc_id]) for vc_id in random_val_classes]

    def __realize_random_parameters(self, parameters : Dict[str, Any]) -> Dict[str, Any]:
        realized_params : Dict[str, Any] = {}

        # There could be some keys that specify a minumum and a maximum constraint for a parameters (example min_span_width, max_span_width)
        # In this case the values that we randomly extract from a minium or a maximum value must be coherent (the max should be >= than the min)
        # In the models these minimum and maximum constraint can be detected by finding two keys that start min_<X> max_<X> where <X> is a string
        # in common
        min_keys = set(k.lstrip("min_") for k in parameters.keys() if k.startswith("min_"))
        max_keys = set(k.lstrip("max_") for k in parameters.keys() if k.startswith("max_"))
        min_max_constrained_parameters = min_keys & max_keys

        # First we insert all the non random keys
        for param_name, param_values in parameters:
            if isinstance(param_values, dict) and "RANDOM" in param_values:
                continue
            realized_params[param_name] = param_values
        # then we extract the random ones
        for param_name, param_values in parameters:
            if isinstance(param_values, dict) and "RANDOM" in param_values:
                # The param contains random values to be extracted
                random_values = param_values["RANDOM"]
                base_param_name = param_name.lstrip("min_").lstrip("max_")
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
                realized_params[param_name] = random_choice(eligible_values)

        return realized_params

    def __get_models(self):
        models : Set[OperatorType] = set()
        for injectable_site in self.__injectable_sites:
            if injectable_site.operator_type not in models:
                models.add(injectable_site.operator_type)
        temp_names = [model.get_model_name() for model in models]
        return temp_names
    
    def __load_models(self, models_folder):
        spatial_models : Dict[str, Any] = {}
        for model_operator_name in self.__get_models():
            spatial_model_path = os.path.join(models_folder, model_operator_name,
                                              f'{model_operator_name}.json')
            with open(spatial_model_path, "r") as spatial_model_json:
                spatial_model = json.load(spatial_model_json)
                if operator_names_table[model_operator_name] not in spatial_models:
                    spatial_models[operator_names_table[model_operator_name]] = {k : v for k, v in spatial_model.items() if not k.startswith("_")}

        return spatial_models




    def __select_corrupted_value(self, model_corrupted_values_domain):
        """
        Selects a domain among the availables (NaN, Zeroes, [-1, 1] and Others) and
        a value from that domain.

        For NaN, Zeroes and Others the value returned is ready to use or to be inserted,
        while for [-1, 1] the value has to be added to the target value.

        Arguments:
            model_corrupted_values_domain {dict} -- Dictionary containing the domains (strings) as keys and
            their probabilities as values.

        Returns:
            [tuple(string, numpy.float32)] -- Returns a tuple that contains the domain and
            a value from that domain.
        """
        # Selects a domain among the NaN, Zeroes, [1, -1] and Others.
        domain = self.__random(*self.__unpack_table(model_corrupted_values_domain))
        if domain == NAN:
            # Returns a F32 NaN.
            return InjectionValue.nan()
        elif domain == ZEROES:
            # Returns a F32 zero.
            return InjectionValue.zeroes() 
        elif domain == BETWEEN_ONE:
            # Returns a F32 between -1 and 1.
            return InjectionValue.between_one(np.random.uniform(low=-1.0, high=1.001))
        elif domain == OTHERS:
            # Returns a 32-long bitstring, interpreted as F32.
            bitstring = "".join(np.random.choice(["0", "1"], size=32))
            integer_bitstring = int(bitstring, base=2)
            float_bitstring = np.frombuffer(np.array(integer_bitstring), dtype=np.float32)[0]
            return InjectionValue.others(float_bitstring)

    def __select_multiple_corrupted_values(self, model_corrupted_values_domain, size):
        """
        Returns multiple pairs of (domain, value) as many as the indicated size.
        It behaves like the similar scalar method.

        Arguments:
            model_corrupted_values_domain {dict} -- Dictionary containing the domains (strings) as keys and
            their probabilities as values.
            size {int} -- Number of tuple to generate.

        Returns:
            [list(tuple(string, numpy.float32))] -- Returns a list of tuple, containing the domain and
            a value from that domain.
        """
        return [self.__select_corrupted_value(model_corrupted_values_domain) for _ in range(size)]


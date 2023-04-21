import os
import json
from collections import OrderedDict
import numpy as np
import operator
import functools
import traceback
import struct
import os.path

from src.operators import OperatorType


class InjectableSite(object):
    """
    Describes an injectable operator and it is characterized by the TensorFlow's operator type,
    the TensorFlow's operator graph name and the size of the output tensor.
    """

    def __init__(self, operator_type, operator_name, size):
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

    @staticmethod
    def nan():
        return InjectionValue("NaN", np.float32(np.nan))

    @staticmethod
    def zeroes():
        return InjectionValue("Zeroes", np.float32(0.0))

    @staticmethod
    def between_one(raw_value):
        return InjectionValue("[-1,1]", np.float32(raw_value))

    @staticmethod
    def others(raw_value):
        return InjectionValue("Others", np.float32(raw_value))

    def __init__(self, value_type, raw_value):
        self.__value_type = value_type
        self.__raw_value = raw_value

    def __str__(self):
        return "({}, {})".format(self.__value_type, hex(struct.unpack('<I', struct.pack('<f', self.__raw_value))[0]))

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

    def get_indexes_values(self):
        return zip(self.__indexes, self.__values)

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

NAN = "NaN"
ZEROES = "Zeroes"
BETWEEN_ONE = "[-1, 1]"
OTHERS = "Others"

SAME_FEATURE_MAP_SAME_ROW = 0
SAME_FEATURE_MAP_SAME_COLUMN = 1
SAME_FEATURE_MAP_BLOCK = 2
SAME_FEATURE_MAP_RANDOM = 3
MULTIPLE_FEATURE_MAPS_BULLET_WAKE = 4
MULTIPLE_FEATURE_MAPS_BLOCK = 5
MULTIPLE_FEATURE_MAPS_SHATTER_GLASS = 6
MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS = 7
MULTIPLE_FEATURE_MAPS_UNCATEGORIZED = 8
MULTIPLE_FEATURE_SKIP_4 = 9
CHANNEL_ALIGNED_BLOCKS = 10
TENSOR_ALIGNED_SINGLE_BLOCK = 11


BLOCK_SIZE = 16


class InjectionSitesGenerator(object):
    def __init__(self, injectable_sites, mode, models_folder='models'):
        self.__injectable_sites = injectable_sites
        self.__cardinalities = self.__load_cardinalities(models_folder)
        self.__corrupted_values_domain = self.__load_corrupted_values_domain(mode, models_folder)
        self.__spatial_models = self.__load_spatial_models(models_folder)
        self.__warp_spatial_models = self.__load_warp_spatial_models(models_folder)
        self.__debugcardinality = -1

    def generate_random_injection_sites(self, size):

        injectables_site_indexes = np.random.choice(len(self.__injectable_sites), size=size)
        injection_sites = []
        cardinalities = []
        patterns = []
        for index in injectables_site_indexes:
            run = True
            while run:
                try:
                    injectable_site = self.__injectable_sites[index]
                    operator_type = injectable_site.operator_type.name
                    if operator_type == 'Conv2D':
                        operator_type = 'Conv2D1x1'
                    if operator_type == 'FusedBatchNorm':
                        operator_type = 'FusedBatchNormV3'
                    cardinality = self.__select_cardinality(self.__cardinalities[operator_type])
                    self.__debugcardinality = cardinality
                    injection_site = InjectionSite(injectable_site.operator_name)
                    corrupted_values = self.__select_multiple_corrupted_values(
                        self.__corrupted_values_domain[operator_type], cardinality)
                    indexes = self.__select_spatial_pattern(self.__spatial_models[operator_type], self.__warp_spatial_models[operator_type], cardinality,
                                                            eval(injectable_site.size), self.__random([0, 1], [0.0, 1.0]))
                    for idx, value in zip(indexes, corrupted_values):
                        injection_site.add_injection(idx, value)
                    injection_sites.append(injection_site)
                    cardinalities.append(self.__debugcardinality)
                    patterns.append(self.__debugspatial_model)
                    run = False
                except Exception as exception:
                    print(exception)
                    traceback.print_exc()
                    continue

        return injection_sites, cardinalities, patterns

    def __get_models(self):
        models = set()
        for injectable_site in self.__injectable_sites:
            if injectable_site.operator_type not in models:
                models.add(injectable_site.operator_type)
        temp_names = [model.get_model_name() for model in models]
        return temp_names

    def __load_cardinalities(self, models_folder):
        """
        Loads the cardinalities for each model.
        It creates a dictionary for each model, containing
        the cardinalities and their probability distribution.

        Returns:
            [dict] -- Returns a dictionary, having the models as keys and the
            cardinalities, associated to each model, as values.
        """
        cardinalities = {}  # Map of cardinalities for each model.
        for model_operator_name in self.__get_models():  # operator_names_table.keys():
            # Folders are named as "SX_model", while files are names "model_SX"
            # So, it is needed to reverse the model name to compose the cardinalities file path,
            separator = model_operator_name.index("_")
            model_prefix = model_operator_name[:separator], model_operator_name[separator + 1:]
            experiment_name = model_prefix[1] + "_" + model_prefix[0]
            base_path = os.path.dirname(os.path.realpath(__file__))
            model_cardinalities_path = base_path + f"/{models_folder}/{model_operator_name}/{experiment_name}" \
                                                   f"_anomalies_count.json "
            # Open the cardinalities file path and load it as a json file.
            with open(model_cardinalities_path, "r") as cardinalities_json:
                model_cardinalities = json.load(cardinalities_json)

                # Add each cardinality model to the map.
                # The insertion is done in order, so keys (the cardinalities) are sorted
                # and converted from string (json) to integer.
                # Only the probability of each cardinality is preserved, the absolute frequency
                # is not relevant for this problem.
                if operator_names_table[model_operator_name] not in cardinalities:
                    cardinalities[operator_names_table[model_operator_name]] = OrderedDict()

                # print(cardinalities)

                for cardinality in sorted(model_cardinalities.keys(), key=lambda x: int(x)):
                    cardinalities[operator_names_table[model_operator_name]][int(cardinality)] = float(
                        model_cardinalities[cardinality][1])

                probabilities_left = 1.0 - sum(cardinalities[operator_names_table[model_operator_name]].values())
                cardinalities[operator_names_table[model_operator_name]][int(cardinality)] += probabilities_left
        return cardinalities

    def __load_corrupted_values_domain(self, mode, models_folder):
        """
        Loads the corrupted values domain for each model.
        It creates a dictionary for each model, containing
        the domains and their probability distribution.

        Returns:
            [dict] -- Returns a dictionary having the models as keys and the
            corrupted values domains, associated to each model, as values.
        """

        def extract_value(line):
            # Each line is composed by an identifier, colon and then the float value.
            # "Identifier: 0.345345"
            # The line is split according the colon, and is selected the component containing
            # the float value, avoiding the last character that is "\n".
            # Then is simply converted to float.
            try:
                value = float(line.split(":")[1][1:])
            except IndexError:
                value = 0.0
            return value

        def get_line(lines, word):
            # Searches the first line, if present, that contains the word parameter.
            # Otherwise, returns an empty string.
            for line in lines:
                if word in line:
                    return line
            return ""

        corrupted_values_domain = {}  # Map of models and their corrupted values' domain.
        for model_operator_name in self.__get_models():
            # The file is simply named as "value_analysis" and is common to each model.
            base_path = os.path.dirname(os.path.realpath(__file__))
            value_path = base_path + f"/{models_folder}/{model_operator_name}/value_analysis.txt"
            with open(value_path, "r") as value_analysis_file:
                # Read the files as text lines.
                model_corrupted_values_domain = OrderedDict()
                lines = value_analysis_file.readlines()
                # Extracts the NaN, Zeroes, [-1, 1] and Others probabilities.
                model_corrupted_values_domain[NAN] = extract_value(get_line(lines, "NaN"))
                model_corrupted_values_domain[ZEROES] = extract_value(get_line(lines, "Zeros"))
                valid_scale_factor = extract_value(get_line(lines, "Valid"))
                model_corrupted_values_domain[BETWEEN_ONE] = extract_value(
                    get_line(lines, "[-1, 1]")) * valid_scale_factor
                model_corrupted_values_domain[OTHERS] = extract_value(get_line(lines, "Others")) * valid_scale_factor
                probability_left = 1.0 - sum(model_corrupted_values_domain.values())
                model_corrupted_values_domain[OTHERS] += probability_left
                # Set the corrupted domain to the relative model.
                corrupted_values_domain[operator_names_table[model_operator_name]] = model_corrupted_values_domain
        return corrupted_values_domain

    def __load_spatial_models(self, models_folder):
        spatial_models = {}
        for model_operator_name in self.__get_models():
            base_path = os.path.dirname(os.path.realpath(__file__))
            spatial_model_path = base_path + f"/{models_folder}/{model_operator_name}/{model_operator_name}" \
                                             f"_spatial_model.json"
            with open(spatial_model_path, "r") as spatial_model_json:
                spatial_model = json.load(spatial_model_json)
                if operator_names_table[model_operator_name] not in spatial_models:
                    spatial_models[operator_names_table[model_operator_name]] = spatial_model
        return spatial_models

    def __load_warp_spatial_models(self, models_folder):
        spatial_models = {}
        for model_operator_name in self.__get_models():
            base_path = os.path.dirname(os.path.realpath(__file__))
            spatial_model_path = base_path + f"/{models_folder}/{model_operator_name}/{model_operator_name}" \
                                             f"_warp_spatial_model.json"
            with open(spatial_model_path, "r") as spatial_model_json:
                spatial_model = json.load(spatial_model_json)
                if operator_names_table[model_operator_name] not in spatial_models:
                    spatial_models[operator_names_table[model_operator_name]] = spatial_model
        return spatial_models

    def __unpack_table(self, table):
        """
        Given a lookup table, implemented as a dictionary, it separates the keys from values
        and returns them in pairs but in different lists.
        Arguments:
            table {dict} -- Lookup table.

        Returns:
            [list, list] -- Returns two lists, the first one contains the keys while the
            latter contains the values.
        """
        keys = []
        values = []
        # Move each pair of key and value to two separate
        # lists to keep the order.
        for key, value in table.items():
            keys.append(key)
            values.append(value)
        return keys, values

    def __random(self, options, probabilities, samples=1):
        """
        Selects one or multiple option(s) according to the probability
        distribution given.

        Arguments:
            options {list} -- List of options.
            probabilities {list(float)} -- List of floats that describes the probability
            distribution associated to options.

        Keyword Arguments:
            samples {int} -- Number of samples to selects. (default: {1})

        Returns:
            [scalar or list] -- Returns a scalar option if samples is 1, otherwise
            returns a list of options.
        """
        # Return a random option, or more than one, according to the probabilities' distribution.
        # The if is needed because specifying size set to 1 returns an array instead of a scalar.
        # In case of samples > 1, is the intended behavior.
        for i in range(len(probabilities)):
            if probabilities[i] < 0.0:
                probabilities[i] = 0.0
        if sum(probabilities) < 1.0:
            remainder = 1.0 - sum(probabilities)
            probabilities[-1] += remainder
        if samples > 1:
            # z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(probabilities), 0, 1)))
            # values, indices = tf.math.top_k(tf.math.log(probabilities) + z, samples)
            # return tf.gather(options, indices).to_list()
            return np.random.choice(options, size=samples, p=probabilities).tolist()
        else:
            # z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(probabilities), 0, 1)))
            # values, indices = tf.math.top_k(tf.math.log(probabilities) + z, samples)
            # return tf.gather(options, indices)
            return np.random.choice(options, p=probabilities)

    def __select_cardinality(self, model_cardinalities):
        """
        Selects a cardinality among the ones provided for the model according to
        the probability distribution associated.

        Arguments:
            model_cardinalities {dict} -- Dictionary containing the integer cardinalities
            and the probabilities as keys.

        Returns:
            {int} -- Returns the drawn cardinality.
        """
        return self.__random(*self.__unpack_table(model_cardinalities))

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

    def __select_spatial_pattern(self, spatial_model, warp_spatial_model, cardinality, output_size, is_warp):
        def multiply_reduce(iterable):
            """
            Given an iterable multiplieas each element
            :param iterable:
            :return: multiplication of each element
            """
            return functools.reduce(operator.mul, iterable, 1)

        def random_same_feature_map(output_size, max_offset, scale_factor, cardinality):
            random_feature_map = np.random.randint(low=0, high=output_size[1])
            feature_map_size = output_size[2] * output_size[3]
            if max_offset * scale_factor >= feature_map_size:
                max_offset = int(feature_map_size / scale_factor) - 1
            random_starting_index = np.random.randint(low=0, high=feature_map_size - max_offset * scale_factor)
            random_starting_index += random_feature_map * feature_map_size
            offsets = np.random.choice(max_offset, replace=False, size=cardinality - 1)
            indexes = [random_starting_index]
            for offset in offsets:
                indexes.append(random_starting_index + offset * scale_factor)
            return [np.unravel_index(index, shape=output_size) for index in indexes]
        def random_warp_pattern(fault_type, output_size, cardinality):
            if fault_type == SAME_FEATURE_MAP_SAME_ROW:
                return random_same_feature_map(output_size, int(patterns["MAX"]), 1, cardinality)
            elif fault_type == SAME_FEATURE_MAP_RANDOM:
                random_feature_map = np.random.randint(low=0, high=output_size[1])
                feature_map_size = output_size[2] * output_size[3]
                indexes = np.random.choice(feature_map_size, size=cardinality, replace=False)
                return [
                    np.unravel_index(index + random_feature_map * feature_map_size, shape=output_size)
                    for index in indexes
                ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_BULLET_WAKE:
                max_feature_map_offset = int(patterns["MAX"])
                # 20 channels
                # MAX 6
                # cardinality = 5
                if max_feature_map_offset >= output_size[1]:
                    max_feature_map_offset = output_size[1] - 1
                # beteewn 0 and 14 
                feature_map_index = np.random.randint(low=0, high=output_size[1] - max_feature_map_offset)
                try:
                    # 1 2 7 11 14
                    feature_map_offsets = np.random.choice(max_feature_map_offset, size=cardinality - 1, replace=False)
                except ValueError:
                    feature_map_offsets = np.random.choice(max_feature_map_offset, size=cardinality - 1, replace=True)
                # [3, 4, 10, 14, 17]
                feature_map_indexes = [feature_map_index]
                for offset in feature_map_offsets:
                    feature_map_indexes.append(feature_map_index + offset)
                feature_map_size = output_size[2] * output_size[3]
                random_index = np.random.randint(low=0, high=feature_map_size)
                return [
                    np.unravel_index(random_index + feature_map_index * feature_map_size, shape=output_size)
                    for feature_map_index in feature_map_indexes
                ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_SHATTER_GLASS:
                max_offsets = [
                    int(patterns["MAX"][0]),
                    int(patterns["MAX"][1]),
                    int(patterns["MAX"][2]),
                    int(patterns["MAX"][3])
                ]
                try:
                    feature_map_indexes = np.random.choice(output_size[1], replace=False, size=max_offsets[0])
                except:
                    feature_map_indexes = np.random.choice(output_size[1], replace=True, size=max_offsets[0])
                np.random.randint(low=0, high=output_size[2] * output_size[3])
                random_feature_map = np.random.choice(feature_map_indexes)
                remainder = cardinality - len(feature_map_indexes)
                choices = list(range(max_offsets[2], max_offsets[3]))
                choices.remove(0)
                offsets = np.random.choice(choices, size=remainder)
                indexes = []
                for feature_map_index in feature_map_indexes:
                    indexes.append(feature_map_index * output_size[2] * output_size[3])
                for offset in offsets:
                    indexes.append(random_feature_map * output_size[2] * output_size[3] + offset)
                indexes = [idx for idx in indexes if idx >= 0]
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                ]
            elif fault_type == MULTIPLE_FEATURE_SKIP_4:

                max_offsets = [
                    int(patterns["MAX"][0]),
                    int(patterns["MAX"][1]),
                ]
                feature_map_size = output_size[2] * output_size[3]
                max_chan_offset, max_index_offset = max_offset
                max_chan_offset = min(max_chan_offset, output_size[1])
                max_index_offset = min(max_index_offset, feature_map_size)
                random_start_index = np.random.randint(low=0, high=feature_map_size - max_index_offset)

                injection_slots_per_channel = max_index_offset // 4
                needed_channels = (cardinality + injection_slots_per_channel - 1) // injection_slots_per_channel

                random_number_of_channels = np.random.choice(low=min(needed_channels, max_chan_offset), high=max(needed_channels, max_chan_offset) + 1)
                random_chan_offsets = sorted(np.random.choice(max_chan_offset, replace=False, size=random_number_of_channels))
                max_random_chan = random_chan_offsets[-1]
                                                  
                random_base_chan = np.random.randint(low=0, high=output_size[1] - max_random_chan)

                n_slots = len(random_chan_offsets) * injection_slots_per_channel
                random_slots = np.random.choice(n_slots, replace=False, size=min(n_slots, cardinality))
                raveled_offsets = [
                    (random_base_chan +
                    random_chan_offsets[(slot // injection_slots_per_channel)]) # Calculate the channel index of the slot
                    * feature_map_size 
                    + (slot % injection_slots_per_channel) * 4
                    + random_start_index #
                    for slot in random_slots
                ]
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in raveled_offsets if index < max_linear_index
                ]
            elif fault_type == CHANNEL_ALIGNED_BLOCKS:
                # Difference of the indices between the first faulty channel and the last faulty channel (capped to the length of output tensor)
                max_chan_offset = int(patterns["MAX"])
                max_chan_offset = min(max_chan_offset, output_size[1])
                # Block alignment
                align = 32
                feature_map_size = output_size[2] * output_size[3]
                # Size of the reminder block (ex. if feature map is 50 and align is 32, will be 18)
                remainder_block_length = align if feature_map_size % align == 0 else feature_map_size % align
                # Check the remainder block is big enough to reach the desired cardinality
                allow_remainder_block = remainder_block_length == align or remainder_block_length * max_chan_offset >= cardinality

                if allow_remainder_block:
                    n_blocks = (feature_map_size + align - 1) // align                    
                else:
                    # Exclude remained block (last block)
                    n_blocks = feature_map_size // align
                # Select block where to inject in all tensors
                random_block = np.random.choice(low=0, high=n_blocks)
                # Calculate size (== align, except for remaineder)
                block_size = min(align, feature_map_size - random_block * align)
                # Minimum channels needed for reaching error
                needed_channels = (cardinality + block_size - 1) // block_size
                # Select number of corrupted channels in the range allowed
                random_number_of_channels = np.random.choice(low=min(needed_channels, max_chan_offset), high=max(needed_channels, max_chan_offset) + 1)
                # Select which channels will be corrupted (starting from the base channel drawn after)
                random_chan_offsets = sorted(np.random.choice(max_chan_offset, replace=False, size=min(needed_channels, max_chan_offset)))
                # Select the distance between the first and the last index of channels
                max_random_chan = random_chan_offsets[-1]
                random_base_chan = np.random.randint(low=0, high=output_size[1] - max_random_chan)
                # Number of available slots where to inject faults
                n_slots = random_number_of_channels * block_size
                # Select faulty slots
                random_slots = np.random.choice(n_slots, replace=False, size=min(n_slots, cardinality))
                # Map slot number to raveled index
                raveled_offsets = [
                    (random_base_chan +
                    random_chan_offsets[(slot // injection_slots_per_channel)]) # Calculate the channel of the slot
                    * feature_map_size 
                    + (slot % injection_slots_per_channel) 
                    + random_block * align #
                    for slot in random_slots
                ]
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in raveled_offsets if index < max_linear_index
                ]
            else:
                indexes = np.random.choice(max_linear_index, size=cardinality, replace=False)
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                ]


                
        def random_pattern(fault_type, output_size, cardinality):
            if fault_type == SAME_FEATURE_MAP_SAME_ROW:
                return random_same_feature_map(output_size, int(patterns["MAX"]), 1, cardinality)
            elif fault_type == SAME_FEATURE_MAP_SAME_COLUMN:
                return random_same_feature_map(output_size, int(patterns["MAX"]), output_size[3], cardinality)
            elif fault_type == SAME_FEATURE_MAP_BLOCK:
                return random_same_feature_map(output_size, int(patterns["MAX"]), 16, cardinality)
            elif fault_type == SAME_FEATURE_MAP_RANDOM:
                random_feature_map = np.random.randint(low=0, high=output_size[1])
                feature_map_size = output_size[2] * output_size[3]
                indexes = np.random.choice(feature_map_size, size=cardinality, replace=False)
                return [
                    np.unravel_index(index + random_feature_map * feature_map_size, shape=output_size)
                    for index in indexes
                ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_BULLET_WAKE:
                max_feature_map_offset = int(patterns["MAX"])
                # 20 channels
                # MAX 6
                # cardinality = 5
                if max_feature_map_offset >= output_size[1]:
                    max_feature_map_offset = output_size[1] - 1
                # beteewn 0 and 14 
                feature_map_index = np.random.randint(low=0, high=output_size[1] - max_feature_map_offset)
                try:
                    # 1 2 7 11 14
                    feature_map_offsets = np.random.choice(max_feature_map_offset, size=cardinality - 1, replace=False)
                except ValueError:
                    feature_map_offsets = np.random.choice(max_feature_map_offset, size=cardinality - 1, replace=True)
                # [3, 4, 10, 14, 17]
                feature_map_indexes = [feature_map_index]
                for offset in feature_map_offsets:
                    feature_map_indexes.append(feature_map_index + offset)
                feature_map_size = output_size[2] * output_size[3]
                random_index = np.random.randint(low=0, high=feature_map_size)
                return [
                    np.unravel_index(random_index + feature_map_index * feature_map_size, shape=output_size)
                    for feature_map_index in feature_map_indexes
                ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_BLOCK:
                max_block_offset = int(patterns["MAX"])
                if max_block_offset * 16 >= max_linear_index:
                    max_block_offset = int(max_linear_index / 16) - 1
                starting_index = np.random.randint(low=0, high=max_linear_index - max_block_offset * 16)
                offsets = np.random.choice(max_block_offset, replace=False, size=cardinality - 1)
                indexes = [starting_index]
                for offset in offsets:
                    indexes.append(starting_index + offset * 16)
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_SHATTER_GLASS:
                max_offsets = [
                    int(patterns["MAX"][0]),
                    int(patterns["MAX"][1]),
                    int(patterns["MAX"][2]),
                    int(patterns["MAX"][3])
                ]
                try:
                    feature_map_indexes = np.random.choice(output_size[1], replace=False, size=max_offsets[0])
                except:
                    feature_map_indexes = np.random.choice(output_size[1], replace=True, size=max_offsets[0])
                np.random.randint(low=0, high=output_size[2] * output_size[3])
                random_feature_map = np.random.choice(feature_map_indexes)
                remainder = cardinality - len(feature_map_indexes)
                choices = list(range(max_offsets[2], max_offsets[3]))
                choices.remove(0)
                offsets = np.random.choice(choices, size=remainder)
                indexes = []
                for feature_map_index in feature_map_indexes:
                    indexes.append(feature_map_index * output_size[2] * output_size[3])
                for offset in offsets:
                    indexes.append(random_feature_map * output_size[2] * output_size[3] + offset)
                indexes = [idx for idx in indexes if idx >= 0]
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS:
                max_offsets = int(patterns["MAX"])
                feature_map_indexes = np.random.choice(output_size[1], replace=False, size=max_offsets)
                np.random.randint(low=0, high=output_size[2] * output_size[3])
                random_feature_map = np.random.choice(feature_map_indexes)
                remainder = cardinality - len(feature_map_indexes)
                choices = range(max_offsets[2], max_offsets[3])
                choices.remove(0)
                offsets = np.random.choice(choices, size=remainder)
                indexes = []
                for feature_map_index in feature_map_indexes:
                    if feature_map_index != random_feature_map:
                        indexes.append(feature_map_index * output_size[2] * output_size[3])
                for offset in offsets:
                    indexes.append(random_feature_map * output_size[2] * output_size[3] + offset)
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                ]
            else:
                indexes = np.random.choice(max_linear_index, size=cardinality, replace=False)
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                ]

        max_linear_index = multiply_reduce(output_size)
        if is_warp:
            if cardinality == 1:
                self.__debugspatial_model = -1
                selected_index = np.random.randint(low=0, high=max_linear_index)
                return [np.unravel_index(selected_index, shape=output_size)]
            else:
                fault_type = self.__random(*self.__unpack_table(warp_spatial_model[str(cardinality)]["FF"]))
                self.__debugspatial_model = int(fault_type)
                patterns = warp_spatial_model[str(cardinality)]["PF"][fault_type]
                fault_type = int(fault_type)
                # The class has only random patterns
                if len(patterns) == 2 and "MAX" in patterns and "RANDOM" in patterns:
                    return random_warp_pattern(fault_type, output_size, cardinality)
                revised_patterns = patterns.copy()
                # Remove max since is not a pattern
                revised_patterns.pop("MAX", None)
                # Extract a pattern from the avaialble one (including random)
                pattern = self.__random(*self.__unpack_table(revised_patterns))
                if pattern == "RANDOM":
                    return random_warp_pattern(fault_type, output_size, cardinality)
                else:
                    pattern = eval(pattern)
                    if fault_type == SAME_FEATURE_MAP_SAME_ROW:
                        # pattern format = (0, ...,max_col) -> Affected columns
                        # even if the pattern is same row, we allow the errors to "spill over" in the next row (it happens with feature maps of uneven sizes)
                        max_offset = pattern[-1]
                        feature_map_size = output_size[2] * output_size[3]
                        assert max_offset <= feature_map_size

                        # Pick a random channel
                        random_channel = np.random.randint(0, output_size[1])
                        random_start_index = np.random.randint(0, feature_map_size - max_offset)

                        indexes = [
                            random_channel * feature_map_size + random_start_index + offset
                            for offset in pattern
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == MULTIPLE_FEATURE_SKIP_4 or fault_type == MULTIPLE_FEATURE_MAPS_SHATTER_GLASS:
                        print(pattern)
                        print(output_size)
                        max_chan_offset = max(channel_offset for channel_offset, feat_map_offsets in pattern)
                        max_feat_map_offset = max(feat_map_offsets[-1] for channel_offset, feat_map_offsets in pattern)
                        # min_feat_map_offset must not be more than zero, by construction
                        min_feat_map_offset = min(feat_map_offsets[0] for channel_offset, feat_map_offsets in pattern)
                        delta_feat_map_offset = max_feat_map_offset - min_feat_map_offset  
                        # Check the applicability of the pattern (enough channels and enough offset)
                        feature_map_size = output_size[2] * output_size[3]
                        assert max_chan_offset <= output_size[1]
                        assert delta_feat_map_offset <= output_size[2] * output_size[3]
                        # Fit the errors keeping in mind possible negative values values then they should fit
                        random_start_index = np.random.randint(0, feature_map_size - delta_feat_map_offset) - min_feat_map_offset
                        # Select the base channel
                        random_channel = np.random.randint(0, output_size[1] - max_chan_offset)
                        indexes = [
                            (random_channel + channel_offset) * feature_map_size + random_start_index + feat_map_offset
                            for channel_offset, feat_map_offsets in pattern for feat_map_offset in feat_map_offsets
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == CHANNEL_ALIGNED_BLOCKS:
                        print(pattern)
                        align, spatial_pattern = pattern
                        max_chan_offset = max(channel_offset for channel_offset, feat_map_offsets in spatial_pattern)
                        feature_map_size = output_size[2] * output_size[3]
                        assert max_chan_offset <= output_size[1]
                        # Round by excess align (even incomplete blocks can be hit by a block error)
                        number_of_blocks = (feature_map_size + align - 1) // align
                        random_block = np.random.randint(0, number_of_blocks)
                        random_start_index = random_block * align
                        random_channel = np.random.randint(0, output_size[1] - max_chan_offset)
                        indexes = [
                            (random_channel + channel_offset) * feature_map_size + random_start_index + feat_map_offset
                            for channel_offset, feat_map_offsets in spatial_pattern for feat_map_offset in feat_map_offsets
                            if 0 < random_start_index + feat_map_offset < feature_map_size 
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == MULTIPLE_FEATURE_MAPS_BULLET_WAKE:
                        max_chan_offset = pattern[-1]
                        assert max_chan_offset <= output_size[1]

                        random_start_channel = np.random.randint(0, output_size[1] - max_chan_offset)
                        feature_map_size = output_size[2] * output_size[3]
                        random_position = np.random.randint(0, feature_map_size)

                        indexes = [
                            (random_start_channel + offset) * feature_map_size + random_position
                            for offset in pattern
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == SAME_FEATURE_MAP_RANDOM:
                        random_feature_map = np.random.randint(0, output_size[1])
                        indexes = np.random.choice(output_size[2] * output_size[3], replace=False, size=cardinality)
                        return [
                            np.unravel_index(index + random_feature_map * output_size[2] * output_size[3],
                                             shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == MULTIPLE_FEATURE_MAPS_UNCATEGORIZED:
                        indexes = np.random.choice(max_linear_index, size=cardinality, replace=False)
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
        else:
            if cardinality == 1:
                self.__debugspatial_model = -1
                selected_index = np.random.randint(low=0, high=max_linear_index)
                return [np.unravel_index(selected_index, shape=output_size)]
            else:
                fault_type = self.__random(*self.__unpack_table(spatial_model[str(cardinality)]["FF"]))
                self.__debugspatial_model = int(fault_type)
                patterns = spatial_model[str(cardinality)]["PF"][fault_type]
                fault_type = int(fault_type)
                if len(patterns) == 2 and "MAX" in patterns and "RANDOM" in patterns:
                    return random_pattern(fault_type, output_size, cardinality)
                revised_patterns = patterns.copy()
                revised_patterns.pop("MAX", None)
                pattern = self.__random(*self.__unpack_table(revised_patterns))
                if pattern == "RANDOM":
                    return random_pattern(fault_type, output_size, cardinality)
                else:
                    pattern = eval(pattern)
                    if fault_type == SAME_FEATURE_MAP_SAME_ROW:
                        assert pattern[-1] <= output_size[2] * output_size[3]
                        np.random.randint(0, output_size[1])
                        random_index = np.random.randint(0, output_size[2] * output_size[3] - pattern[-1])
                        indexes = [
                            random_index + offset
                            for offset in pattern
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == SAME_FEATURE_MAP_SAME_COLUMN:
                        assert pattern[-1] <= output_size[3]
                        np.random.randint(0, output_size[1])
                        random_index = np.random.randint(0, output_size[2] * output_size[3] - pattern[-1] * output_size[3])
                        indexes = [
                            random_index + offset * output_size[3]
                            for offset in pattern
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == SAME_FEATURE_MAP_BLOCK:
                        assert pattern[-1] * 16 <= output_size[2] * output_size[3]
                        np.random.randint(0, output_size[1])
                        random_index = np.random.randint(0, output_size[2] * output_size[3] - pattern[-1] * 16)
                        indexes = [
                            random_index + offset * output_size[3]
                            for offset in pattern
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == SAME_FEATURE_MAP_RANDOM:
                        random_feature_map = np.random.randint(0, output_size[1])
                        indexes = np.random.choice(output_size[2] * output_size[3], replace=False, size=cardinality)
                        return [
                            np.unravel_index(index + random_feature_map * output_size[2] * output_size[3],
                                             shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == MULTIPLE_FEATURE_MAPS_BULLET_WAKE:
                        if pattern[-1] >= output_size[1]:
                            new_card = 0
                            for elem in pattern:
                                if elem < output_size[1]:
                                    new_card += 1
                            pattern = pattern[:new_card]
                            self.__debugcardinality = new_card
                        starting_feature_map_index = np.random.randint(0, output_size[1] - pattern[-1])
                        random_index = np.random.randint(0, output_size[2] * output_size[3])
                        indexes = [
                            random_index + (starting_feature_map_index + offset) * output_size[2] * output_size[3]
                            for offset in pattern
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == MULTIPLE_FEATURE_MAPS_BLOCK:
                        if max_linear_index < 16 * pattern[-1]:
                            new_card = 0
                            for elem in pattern:
                                if max_linear_index > elem * 16:
                                    new_card += 1
                            pattern = pattern[:new_card]
                            self.__debugcardinality = new_card
                        random_index = np.random.randint(0, max_linear_index - 16 * pattern[-1])
                        indexes = [
                            random_index + 16 * offset
                            for offset in pattern
                        ]
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif (
                            fault_type == MULTIPLE_FEATURE_MAPS_SHATTER_GLASS or
                            fault_type == MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS
                    ):
                        if pattern[-1][0] >= output_size[1]:
                            new_card = 0
                            for elem in pattern:
                                if elem[0] < output_size[1]:
                                    new_card += 1
                            pattern = pattern[:new_card]
                        assert pattern[-1][0] < output_size[1]
                        min_x = 0
                        max_x = 0
                        for feature in pattern:
                            if feature[1][0] < min_x:
                                min_x = feature[1][0]
                            if feature[1][-1] > max_x:
                                max_x = feature[1][-1]
                        random_feature_map = np.random.randint(0, output_size[1] - pattern[-1][0])
                        random_index = np.random.randint(min_x + 1, output_size[2] * output_size[3] - max_x)
                        indexes = []
                        for feature_pattern in pattern:
                            for offset in feature_pattern[1]:
                                indexes.append(
                                    random_index + offset + (random_feature_map + feature_pattern[0]) * output_size[2] *
                                    output_size[3])
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]
                    elif fault_type == MULTIPLE_FEATURE_MAPS_UNCATEGORIZED:
                        indexes = np.random.choice(max_linear_index, size=cardinality, replace=False)
                        return [
                            np.unravel_index(index, shape=output_size)
                            for index in indexes
                        ]

import math
from typing import Any, Dict, List, Optional
import numpy as np

from pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range


def skip_4_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"])
    skip_amount = params["skip_amount"]
    unique_map_indexes = params["unique_channel_indexes"]

    remainder = np.random.randint(0, skip_amount)
    max_starting_map_offset = 1 + int(math.floor((num_values_per_channel - remainder - skip_amount * unique_map_indexes) / skip_amount))
    starting_map_offset = np.random.randint(0, max_starting_map_offset)

    candidate_corrupt_positions = [(chan, map_idx + starting_map_offset * skip_amount + remainder) for chan in channels for map_idx in range(unique_map_indexes)]
    number_of_corrupted_pos = random_int_from_pct_range(len(candidate_corrupt_positions), *params["indexes_corruption_pct"])
    corrupt_positions = np.random.choice(candidate_corrupt_positions, number_of_corrupted_pos)
    return convert_to_linearized_index(corrupt_positions, output_shape)
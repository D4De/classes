from typing import List, Dict, Any, Optional

import numpy as np

from src.pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range


def bullet_wake_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"], min_channels=2)
    random_position = np.random.randint(0, num_values_per_channel)
    corrupt_positions = [(chan, random_position) for chan in channels]
    return convert_to_linearized_index(corrupt_positions, output_shape)
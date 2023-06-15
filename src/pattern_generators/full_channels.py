from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from src.pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range


def full_channels_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"])
    corrupted_positions = []
    avg_chan_corruption_pct : Tuple[float, float] = params["avg_channel_corruption_pct"]
    for chan in channels:
        num_corr_positions = random_int_from_pct_range(num_values_per_channel, *avg_chan_corruption_pct)
        positions = np.random.choice(num_corr_positions, num_values_per_channel)
        corrupted_positions += [(chan, pos) for pos in positions]
    return convert_to_linearized_index(corrupted_positions, output_shape)
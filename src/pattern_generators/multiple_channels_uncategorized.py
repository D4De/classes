from typing import List, Dict, Any, Optional

import numpy as np
from src.pattern_generators.generator_utils import clamp, convert_to_linearized_index, random_channels, random_int_from_pct_range


def multiple_channels_uncategorized_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"], min_channels=2)

    corrupted_values = set()

    for channel in channels:
        num_channel_corrupted_values = random_int_from_pct_range(num_values_per_channel, *params["avg_channel_corruption_pct"])
        num_channel_corrupted_values = clamp(num_channel_corrupted_values, params["min_errors_per_channel"], params["max_errors_per_channel"])
        channel_corr_pos = np.random.choice(num_values_per_channel, num_channel_corrupted_values, replace=False)
        for pos in channel_corr_pos:
            corrupted_values.add((channel, pos))

    return convert_to_linearized_index(corrupted_values, output_shape)
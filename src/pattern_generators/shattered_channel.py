from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from src.pattern_generators.generator_utils import clamp, convert_to_linearized_index, random_channels, random_int_from_pct_range


def shattered_channel_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"], min_channels=2)
    
    common_position = np.random.randint(0, num_values_per_channel)


    corrupted_values = set()

    for channel in channels:
        span_width = clamp(np.random.randint(params["min_span_width"], params["max_span_width"] + 1), 1, num_values_per_channel)
        span_begin = np.random.randint(max(0, common_position - span_width), max(min(num_values_per_channel - 1, num_values_per_channel - span_width), 1))
        channel_num_corr_pos = random_int_from_pct_range(span_width, *params["avg_span_corruption_pct"])
        channel_corr_pos = np.random.choice(span_width, channel_num_corr_pos, replace=False)
        for pos in channel_corr_pos:
            corrupted_values.add((channel, span_begin + pos))
        corrupted_values.add((channel, common_position))

    return convert_to_linearized_index(corrupted_values, output_shape)
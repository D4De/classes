from typing import List, Dict, Any, Optional

import numpy as np

from pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range


def rectangles_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"])

    rectangle_width = params["rectangle_width"]
    rectangle_height = params["rectangle_height"]
    channel_height = output_shape[2]
    channel_width = output_shape[3]

    if channel_height < rectangle_height or channel_width < rectangle_width:
        return None

    random_top = np.random.randint(0, channel_height - rectangle_height)
    random_left = np.random.randint(0, channel_width - rectangle_width)

    top_left_position = random_top * channel_width + random_left

    rectangle_positions = [top_left_position + h * rectangle_width + w for h in range(rectangle_height) for w in range(rectangle_width)]

    corrupted_positions = [(chan, pos) for chan in channels for pos in rectangle_positions]
    
    return convert_to_linearized_index(corrupted_positions, output_shape)
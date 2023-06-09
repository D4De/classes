import logging
import math
from typing import List, Dict, Any, Tuple

import numpy as np

from pattern_generators.utils import convert_to_linearized_index, random_channels, random_int_from_pct_range

logger = logging.getLogger(__name__)

def single_channel_alternated_block_generator(output_shape : List[int], params : Dict[str, Any]) -> List[int]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]

    block_size : int = params["block_size"]
    max_feature_map_width = params["max_feature_map_width"]
    max_feature_map_height = params["max_feature_map_height"]
    min_block_skip = params["min_block_skip"]
    max_block_skip = params["max_block_skip"]

    if max_feature_map_width < 64 and output_shape[3] >= max_feature_map_width:
        return None

    if max_feature_map_height < 64 and output_shape[2] >= max_feature_map_height:
        return None   

    random_channel = np.random.randint(0, num_channels)

    num_blocks = int(math.ceil(num_values_per_channel / 2))
    curr_block = 0

    corrupted_positions = []

    while curr_block < num_blocks:
        corrupted_positions += [(random_channel, i + curr_block * block_size) for i in range(block_size)]
        block_skip = np.random.randint(min_block_skip, max_block_skip + 1)
        curr_block += block_skip


    return convert_to_linearized_index(corrupted_positions, output_shape)
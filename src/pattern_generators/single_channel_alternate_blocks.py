import logging
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from src.pattern_generators.generator_utils import convert_to_linearized_index

logger = logging.getLogger(__name__)

def single_channel_alternated_block_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]

    block_size : int = params["block_size"]
    max_feature_map_size = params["max_feature_map_size"]
    min_block_skip = params["min_block_skip"]
    max_block_skip = params["max_block_skip"]

    random_channel = np.random.randint(0, num_channels)

    num_blocks = int(math.ceil(num_values_per_channel / 2))
    curr_block = 0

    corrupted_positions = []

    while curr_block < num_blocks:
        corrupted_positions += [(random_channel, i + curr_block * block_size) for i in range(block_size)]
        block_skip = np.random.randint(min_block_skip, max_block_skip + 1)
        curr_block += block_skip


    return convert_to_linearized_index(corrupted_positions, output_shape)
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from src.pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range

logger = logging.getLogger(__name__)

def multiple_channels_multi_block_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"], min_channels=2)
    corrupted_positions = []
    avg_block_corruption_pct : Tuple[float, float] = params["avg_block_corruption_pct"]
    block_size : int = params["block_size"]

    num_blocks_per_channel = num_values_per_channel // block_size
    # Consider the remainder block valid only if is at least half of the normal block length
    if num_values_per_channel % block_size >= block_size // 2:
        num_blocks_per_channel += 1
        remainder_block_included = True
    else:
        remainder_block_included = False
    
    if num_blocks_per_channel == 0:
        logger.warn(f"Failed to inject multi_channels_multi_block: Block of {block_size} too big for channel size ({output_shape[2]}x{output_shape[3]})")
        return None
    
    random_block = np.random.randint(0, num_blocks_per_channel)
    # picked_block_suze contains the real block size of the selected block
    # It is equal block_size unless the block is a remainder
    if random_block == num_blocks_per_channel - 1 and remainder_block_included:
        picked_block_size = num_values_per_channel % block_size
    else:
        picked_block_size = block_size       

    for chan in channels:
        
        num_corr_positions = max(block_size // 2, random_int_from_pct_range(picked_block_size, *avg_block_corruption_pct))
        positions = np.random.choice(picked_block_size, num_corr_positions)
        corrupted_positions += [(chan, block_size * random_block + pos) for pos in positions]

    return convert_to_linearized_index(corrupted_positions, output_shape)
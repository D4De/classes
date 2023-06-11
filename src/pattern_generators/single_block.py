import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range

logger = logging.getLogger(__name__)

def single_block_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    num_values_per_tensor = num_channels * num_values_per_channel
    block_corruption_pct : Tuple[float, float] = params["block_corruption_pct"]
    block_size : int = params["block_size"]

    

    block_start_offset = np.random.randint(0, num_values_per_tensor - block_size)
    cardinality = random_int_from_pct_range(block_size, *block_corruption_pct)
    block_corrupted_idxs = np.random.choice(block_size, cardinality)
    corrupted_positions = [block_start_offset + idx for idx in block_corrupted_idxs]

    return corrupted_positions
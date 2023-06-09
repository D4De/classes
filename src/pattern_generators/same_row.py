import logging
import math
from typing import List, Dict, Any, Tuple

import numpy as np

from pattern_generators.utils import convert_to_linearized_index, random_channels, random_int_from_pct_range

logger = logging.getLogger(__name__)

def same_row_generator(output_shape : List[int], params : Dict[str, Any]) -> List[int]:
    
    num_channels = output_shape[1]
    row_corruption_pct = params["row_corruption_pct"]
    max_cardinality = params["max_cardinality"]
    min_value_skip = params["min_value_skip"]
    max_value_skip = params["max_value_skip"]
    num_rows = output_shape[2]
    num_cols = output_shape[3]

    rows_indexes = random_channels(num_cols, min_value_skip, max_value_skip, max_cardinality, *row_corruption_pct)

    random_channel = np.random.randint(0, num_channels)
    random_row = np.random.randint(0, num_rows)

    corrupted_positions = [(random_channel, random_row * num_cols + idx) for idx in rows_indexes]


    return convert_to_linearized_index(corrupted_positions, output_shape)
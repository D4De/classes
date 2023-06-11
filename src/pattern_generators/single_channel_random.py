import logging
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range

logger = logging.getLogger(__name__)

def single_channel_random_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channel_corruption_pct = params["channel_corruption_pct"]
    max_cardinality = params["max_cardinality"]
    min_value_skip = params["min_value_skip"]
    max_value_skip = params["max_value_skip"]

    chan_positions = random_channels(num_values_per_channel, min_value_skip, max_value_skip, max_cardinality, *channel_corruption_pct)

    random_channel = np.random.randint(0, num_channels)

    corrupted_positions = [(random_channel, idx) for idx in chan_positions]


    return convert_to_linearized_index(corrupted_positions, output_shape)
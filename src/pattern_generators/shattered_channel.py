from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range


def shattered_channel_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    channels = random_channels(num_channels, params["min_channel_skip"], params["max_channel_skip"], params["max_corrupted_channels"], *params["affected_channels_pct"])
    pass
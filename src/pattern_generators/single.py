import math
from typing import Any, Dict, List, Optional
import numpy as np

from src.pattern_generators.generator_utils import convert_to_linearized_index, random_channels, random_int_from_pct_range


def single_generator(output_shape : List[int], params : Dict[str, Any]) -> Optional[List[int]]:
    
    num_channels = output_shape[1]
    num_values_per_channel = output_shape[2] * output_shape[3]
    num_values_per_tensor = num_channels * num_values_per_channel
    value = np.random.randint(0, num_values_per_tensor)
    return [value]
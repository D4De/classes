from typing import Any, Callable, Dict, List, Optional
from bullet_wake import bullet_wake_generator
from multiple_channels_multi_block import multiple_channels_multi_block
from same_column import same_row_generator
from shattered_channel import shattered_channel_generator
from single_channel_alternate_blocks import single_channel_alternated_block_generator
from single import single_generator
from full_channels import full_channels_generator
from rectangles import rectangles_generator
from same_row import same_row_generator
from single_block import single_block_generator
from single_channel_random import single_channel_random_generator
from skip_4 import skip_4_generator



generator_functions : Dict[str, Callable[[List[int], Dict[str, Any]], Optional[List[int]]]] = {
    "bullet_wake": bullet_wake_generator,
    "multiple_channels_multi_block": multiple_channels_multi_block,
    "same_column": same_row_generator,
    "shattered_channel": shattered_channel_generator,
    "single_channel_alternate_blocks": single_channel_alternated_block_generator,
    "single": single_generator,
    "full_channels": full_channels_generator,
    "rectangles": rectangles_generator,
    "same_row": same_row_generator,
    "single_block": single_block_generator,
    "single_channel_random": single_channel_random_generator,
    "skip_4": skip_4_generator
}

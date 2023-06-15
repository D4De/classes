from typing import Any, Callable, Dict, List, Optional
from src.pattern_generators.bullet_wake import bullet_wake_generator
from src.pattern_generators.multiple_channels_multi_block import multiple_channels_multi_block_generator
from src.pattern_generators.same_column import same_row_generator
from src.pattern_generators.shattered_channel import shattered_channel_generator
from src.pattern_generators.single_channel_alternate_blocks import single_channel_alternated_block_generator
from src.pattern_generators.single import single_generator
from src.pattern_generators.full_channels import full_channels_generator
from src.pattern_generators.rectangles import rectangles_generator
from src.pattern_generators.same_row import same_row_generator
from src.pattern_generators.single_block import single_block_generator
from src.pattern_generators.single_channel_random import single_channel_random_generator
from src.pattern_generators.skip_4 import skip_4_generator



generator_functions : Dict[str, Callable[[List[int], Dict[str, Any]], Optional[List[int]]]] = {
    "bullet_wake": bullet_wake_generator,
    "multi_channel_block": multiple_channels_multi_block_generator,
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

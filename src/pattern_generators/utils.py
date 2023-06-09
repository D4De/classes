import math
from typing import List
import numpy as np

def random_int_from_pct_range(number : int, range_min_pct : float, range_max_pct : float) -> int:
    min_number = int(round(number * range_min_pct / 100.0))
    max_number = int(round(number * range_max_pct / 100.0))

    return np.random.randint(min_number, max_number + 1)

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def random_list_with_gap_constraints(length : int, max_number : int, min_gap : int, max_gap : int) -> List[int]:
    gap_list = [min_gap] * (length - 1)
    head = 0
    tail = min_gap * (length - 1)

    while tail < max_number:
        incrementable_gaps = [i for i, gap in enumerate(gap_list) if gap < max_gap]
        if len(incrementable_gaps) == 0:
            return incrementable_gaps
        random_idx = np.random.choice(incrementable_gaps, 1)
        gap_list[random_idx] += 1
        tail += 1
    
    gap_list = [head]
    accumulator = head
    for gap in gap_list:
        accumulator += gap
        gap_list.append(accumulator)

    return accumulator


def random_channels(num_channels : int, min_channel_skip : int, max_channel_skip : int, max_corrupted_channels : int, corrupted_chan_min_pct : float, corrupted_chan_max_pct : float) -> List[int]:

    max_channels_for_gaps = int(math.floor(num_channels, min_channel_skip)) + 1
    num_corrupted_channels = random_int_from_pct_range(num_channels, corrupted_chan_min_pct, corrupted_chan_max_pct)
    
    num_corrupted_channels = min(num_corrupted_channels, max_channels_for_gaps, max_corrupted_channels)

    min_span = min_channel_skip * (num_corrupted_channels - 1)
    max_span = max_channel_skip * (num_corrupted_channels - 1)
    max_starting_channel = max(num_channels - min_span, 0)

    starting_channel_offset = np.random.randint(0, max_starting_channel)

    channels = random_list_with_gap_constraints(num_corrupted_channels, min(max_span, num_channels), min_channel_skip, max_channel_skip)

    return [idx + starting_channel_offset for idx in channels if (idx + starting_channel_offset) < num_channels]


def convert_to_linearized_index(pos_list : List[int, int], output_shape : List[int]) -> List[int]:
    num_values_per_channel = output_shape[2] * output_shape[3]
    num_values_per_tensor = output_shape[0]
    return [
                np.unravel_index(chan * num_values_per_channel + position, shape=output_shape)
                    for chan, position in pos_list
                    if position < num_values_per_channel
                    if chan * num_values_per_channel + position < num_values_per_tensor
                ]

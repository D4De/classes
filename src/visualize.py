from math import sqrt
import math
import os
from typing import Iterable, Literal, Tuple, Union


import matplotlib.pyplot as plt
import matplotlib

import numpy as np

levels = [0, 1, 2, 3, 4, 5]
labels = [""] + ["nan", "zero", "in_range", "out_of_range"]
colors = ["white", "black", "red", "orange", "brown"]
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="neither")


def split_two(num: int) -> Tuple[int, int]:
    """
    Given a number [num] of plots returns the optimal arrangment for displaying
    the subplot in a 2D grid.
    """
    val = int(math.ceil(sqrt(num)))
    return val, val


def is_square(apositiveint: int) -> bool:
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


def visualize(
    tensor_diff: np.ndarray,
    faulty_channels: Iterable[int],
    layout_type: Literal['NCHW', 'NHWC'],
    output_path: Union[str, None] = None,
    save: bool = False,
    show: bool = True,
    invalidate: bool = False,
    suptitile: str = "",
):
    scene_dim_x, scene_dim_y = split_two(len(faulty_channels))

    if not invalidate and os.path.exists(output_path):
        return

    fig, axs = plt.subplots(scene_dim_x, scene_dim_y)
    if len(suptitile) > 0:
        plt.suptitle(suptitile)

    for i, curr_C in enumerate(faulty_channels):
        if layout_type == 'NCHW':
            slice_diff = tensor_diff[0, curr_C, :, :]
        else:
            slice_diff = tensor_diff[0, :, :, curr_C]
        if len(faulty_channels) == 1:
            # Single Plot
            curr_axs = axs
        elif len(axs.shape) == 1:
            # Multiple plots arranged in a line
            curr_axs = axs[i]
        else:
            # Plots arranged in a 2D Grid
            curr_axs = axs[i % scene_dim_x, i // scene_dim_x]

        # Show image with diff
        img = curr_axs.imshow(slice_diff, cmap=cmap, norm=norm, interpolation="nearest")
        # Clear Axis
        curr_axs.set_yticks([])
        curr_axs.set_xticks([])
        curr_axs.set_yticklabels([])
        curr_axs.set_xticklabels([])
        # Label
        curr_axs.set_title(f"CH {curr_C}", fontsize=9)

    # Add colorbar legend
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.ax.set_yticklabels(labels)

    if show:
        plt.show()

    if save:
        if output_path is None:
            raise ValueError("Output path is required for saving file")

    plt.savefig(output_path)
    plt.close()

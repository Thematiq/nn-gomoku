import torch
import numpy as np

from enum import Enum


class Position(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    LEFT_SIDED_DIAGONAL = 2
    RIGHT_SIDED_DIAGONAL = 3


def create_filter(size: int, n: int, position: Position) -> torch.Tensor:
    """
    Create a filter to be compatible with Convolution Evaluation class
    :param size: Size ouf output, should be equal or greater than size of filter
    :param n: Number of stones per line
    :param position: Weather to check for n elements diagonally, horizontally or vertically
    :return: Single filter that can be passes to torch.nn.functional.conv2d as weight
    """
    filter = torch.zeros((n, n))
    center = np.floor(n / 2.).astype(np.int32)

    x, y, x_update, y_update = None, None, None, None
    if position == Position.VERTICAL:
        x = 0
        y = center
        x_update = 1
        y_update = 0
    elif position == Position.HORIZONTAL:
        x = center
        y = 0
        x_update = 0
        y_update = 1
    elif position == Position.RIGHT_SIDED_DIAGONAL:
        x = 0
        y = 0
        x_update = 1
        y_update = 1
    elif position == Position.LEFT_SIDED_DIAGONAL:
        x = 0
        y = n - 1
        x_update = 1
        y_update = -1

    for _ in range(n):
        filter[x, y] = 1
        x += x_update
        y += y_update

    if n != size:
        proper_size_filter = torch.zeros((size, size))
        proper_size_filter[:n, :n] = filter
        filter = proper_size_filter
    return torch.reshape(filter, (1, 1, filter.shape[0], filter.shape[1]))


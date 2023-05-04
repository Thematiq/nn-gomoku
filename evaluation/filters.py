import torch
import numpy as np

from enum import Enum


class LineBlocked(Enum):
    NO = 0
    BOTH = 1
    UP = 2
    DOWN = 3


class Position(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    LEFT_SIDED_DIAGONAL = 2
    RIGHT_SIDED_DIAGONAL = 3


def create_filter(size: int, n: int, position: Position, corners: LineBlocked) -> torch.Tensor:
    """
    Create a filter to be compatible with Convolution Evaluation class
    :param size: Size ouf output, should be equal or greater than size of filter
    :param n: Number of stones per line
    :param position: Weather to check for n elements diagonally, horizontally or vertically
    :param corners: Check if opponent placed a stone near our line
    :return: Single filter that can be passes to torch.nn.functional.conv2d as weight
    """
    first_negative = 1 if corners == LineBlocked.BOTH or corners == LineBlocked.UP else 0
    last_negative = 1 if corners == LineBlocked.BOTH or corners == LineBlocked.DOWN else 0
    elements = n + first_negative + last_negative
    filter = torch.zeros((elements, elements))

    center = np.floor(elements / 2.).astype(np.int32)

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
        y = elements - 1
        x_update = 1
        y_update = -1

    if first_negative == 1:
        filter[x, y] = -1
        x += x_update
        y += y_update
    for _ in range(n):
        filter[x, y] = 1
        x += x_update
        y += y_update
    if last_negative == 1:
        filter[x, y] = -1

    if elements != size:
        proper_size_filter = torch.zeros((size, size))
        proper_size_filter[:elements, :elements] = filter
        filter = proper_size_filter
    return torch.reshape(filter, (1, 1, filter.shape[0], filter.shape[1]))


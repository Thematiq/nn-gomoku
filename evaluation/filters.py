import torch
import numpy as np

from enum import Enum
from typing import Tuple


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


# TODO finish this function
def create_filter(size: int, n: int, position: Position, corners: LineBlocked) -> torch.Tensor:
    """
    Create a filter to be compatible with Convolution Evaluation class
    :param size:
    :param n:
    :param position:
    :param corners: Determine if
    :return: single filter that can be passes to torch.nn.functional.conv2d as weight
    """
    filter = torch.zeros((size, size))
    center = np.ceil(size / 2.)
    x_start = center - np.ceil(size / 2.)
    y_start = center - np.ceil(size / 2.)

    return filter


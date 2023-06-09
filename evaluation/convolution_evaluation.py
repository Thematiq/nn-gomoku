import torch.nn.functional as F

from .evaluation import Evaluation
from .filters import *


class ConvolutionEvaluation(Evaluation):
    """
    Main concept of conv evaluation:
    Let's suppose that we want to discover 4 stones in a row, to do this we can use following filter:
    [1, 1, 1, 1] and we got value 4 detect a 4 stones row. But we have to dealt also with situation
    when we don't have 4 stones in a row example:
        [1, 1, 0, 1] - we got 3 and didn't have any important feature - using mask_values we can filter this value to 0
        [1, 1, 1, -1] - we got 2 and using mask_filter reduce this value to 0
        [-1, -1, -1, -1] - here we got -4 and have 4 stones but with opposite color - we can keep this information
    """
    def __init__(self, filters, mask_values):
        """
        :param filters: list of filters to be applied to torch.nn.functional as weight parameter
        :param mask_values: list of absolute values - we will zero values with lower absolute value than this.
        """
        self.filters: torch.Tensor = filters
        self.mask_values: torch.Tensor = mask_values

    def evaluate(self, state, player) -> float:
        state = torch.tensor(state, dtype=torch.float32)
        state = state.view(1, 1, state.shape[0], state.shape[1])

        x = F.conv2d(state, self.filters, padding='same')
        x[torch.abs(x) <= self.mask_values] = 0.

        # checking for infinity
        x[x >= 5] = torch.inf
        x[x <= -5] = -1*torch.inf

        x *= x*x

        return float(torch.sum(x))

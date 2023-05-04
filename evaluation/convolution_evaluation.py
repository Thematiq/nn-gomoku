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
        state = self.to_state_tensor(state)

        x = F.conv2d(state, self.filters, padding='same')
        x = torch.where(torch.abs(x) >= self.mask_values, x, 0)

        # checking for infinity
        x[x >= 5] = torch.inf
        x[x <= -5] = -1*torch.inf

        return float(torch.sum(x))

    def to_state_tensor(self, state) -> torch.Tensor:
        """
        Helper function to convert state from 0, 1, 2 to 0, 1, -1
        :param state: Current game state
        :return: Modified state
        """
        tensor_state = torch.tensor(state)
        tensor_state[tensor_state == 2] = -1
        tensor_state = tensor_state.to(torch.float32)
        return torch.reshape(tensor_state, (1, 1, tensor_state.shape[0], tensor_state.shape[1]))

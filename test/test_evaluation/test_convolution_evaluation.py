import torch
import pytest
import numpy as np
import math

from evaluation import create_filter, Position, ConvolutionEvaluation


@pytest.mark.parametrize("state, expected_evaluation", [(np.array([[1, 0, 0, 0, 0],
                                                                   [1, 0, 0, 0, 0],
                                                                   [1, 0, 0, 0, 0],
                                                                   [1, 0, 0, 0, 0],
                                                                   [1, 0, 0, 0, 0]]), math.inf),
                                                        (np.array([[1, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0],
                                                                   [0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 1]]), 0),
                                                        (np.array([[0, 0, 0, 0, 2],
                                                                   [0, 0, 0, 0, 2],
                                                                   [0, 0, 0, 0, 2],
                                                                   [0, 0, 0, 0, 2],
                                                                   [0, 0, 0, 0, 2]]), -1 * math.inf),
                                                        (np.array([[0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0]]), 6)])
def test_evaluation(state, expected_evaluation):
    filters = torch.concatenate([create_filter(5, 5, Position.VERTICAL),
                                 create_filter(5, 3, Position.VERTICAL)])

    evaluator = ConvolutionEvaluation(filters, torch.tensor([[[5.]], [[3.]]]))
    evaluation = evaluator.evaluate(state, None)
    assert evaluation == expected_evaluation

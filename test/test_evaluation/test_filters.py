import torch
import pytest

from evaluation import create_filter, Position


@pytest.mark.parametrize("size, N, position, expected", [
    (5, 3, Position.VERTICAL, torch.tensor([[0, 1, 0, 0, 0],
                                            [0, 1, 0, 0, 0],
                                            [0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]], dtype=torch.float32)),
    (4, 3, Position.VERTICAL, torch.tensor([[0, 1, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 0]], dtype=torch.float32)),
    (5, 5, Position.HORIZONTAL, torch.tensor([[0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0],
                                              [1, 1, 1, 1, 1],
                                              [0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0]], dtype=torch.float32)),
    (6, 4, Position.HORIZONTAL, torch.tensor([[0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [1, 1, 1, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0]], dtype=torch.float32)),
    (5, 4, Position.RIGHT_SIDED_DIAGONAL, torch.tensor([[1, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0],
                                                        [0, 0, 1, 0, 0],
                                                        [0, 0, 0, 1, 0],
                                                        [0, 0, 0, 0, 0]], dtype=torch.float32)),
    (3, 2, Position.LEFT_SIDED_DIAGONAL, torch.tensor([[0, 1, 0],
                                                       [1, 0, 0],
                                                       [0, 0, 0]], dtype=torch.float32))
])
def test_filters(size, N, position, expected):
    results = create_filter(size, N, position)
    assert torch.all(expected == results)

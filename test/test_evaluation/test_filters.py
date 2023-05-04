import torch
import pytest

from evaluation import create_filter, Position, LineBlocked


@pytest.mark.parametrize("size, N, position, blocked, expected", [
    (5, 3, Position.VERTICAL, LineBlocked.NO, torch.tensor([[0, 1, 0, 0, 0],
                                                            [0, 1, 0, 0, 0],
                                                            [0, 1, 0, 0, 0],
                                                            [0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0]], dtype=torch.float32)),
    (4, 2, Position.VERTICAL, LineBlocked.BOTH, torch.tensor([[0, 0, -1, 0],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 1, 0],
                                                              [0, 0, -1, 0]], dtype=torch.float32)),
    (5, 3, Position.HORIZONTAL, LineBlocked.BOTH, torch.tensor([[0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0],
                                                                [-1, 1, 1, 1, -1],
                                                                [0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0]], dtype=torch.float32)),
    (6, 4, Position.HORIZONTAL, LineBlocked.NO, torch.tensor([[0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0],
                                                              [1, 1, 1, 1, 0, 0],
                                                              [0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0]], dtype=torch.float32)),
    (5, 4, Position.RIGHT_SIDED_DIAGONAL, LineBlocked.UP, torch.tensor([[-1, 0, 0, 0, 0],
                                                                        [0, 1, 0, 0, 0],
                                                                        [0, 0, 1, 0, 0],
                                                                        [0, 0, 0, 1, 0],
                                                                        [0, 0, 0, 0, 1]], dtype=torch.float32)),
    (3, 2, Position.LEFT_SIDED_DIAGONAL, LineBlocked.DOWN, torch.tensor([[0, 0, 1],
                                                                         [0, 1, 0],
                                                                         [-1, 0, 0]], dtype=torch.float32))
])
def test_filters(size, N, position, blocked, expected):
    results = create_filter(size, N, position, blocked)
    assert torch.all(expected == results)

import numpy as np
import pytest

from agents import AlphaBetaAgent
from evaluation import ConvolutionEvaluation, create_check_final_filter

first_board = np.array([[1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, -1, -1, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, -1, 0, 0, -1]])
second_board = np.array([[1, 0, 1, 1, 0],
                         [0, 1, 0, 0, -1],
                         [0, -1, 0, 0, -1],
                         [-1, 0, 0, 1, 0],
                         [0, -1, 0, -1, 1]])


def act_on_board(board, opponent, depth):
    board = np.copy(board)
    if opponent:
        board *= -1
    agent = AlphaBetaAgent(depth=depth, evaluator=ConvolutionEvaluation(*create_check_final_filter()))
    return agent.act(board, opponent)


@pytest.mark.parametrize("board, expected_result", [(first_board, 4), (second_board, 12)])
@pytest.mark.parametrize("opponent", [True, False])
def test_found_best_move(board, expected_result, opponent):
    move = act_on_board(board, opponent, 1)
    assert move == expected_result


@pytest.mark.parametrize("board, expected_result", [(first_board, 4), (second_board, 12)])
@pytest.mark.parametrize("opponent", [True, False])
def test_blocking_opponent(board, expected_result, opponent):
    move = act_on_board(board, opponent, 2)
    assert move == expected_result

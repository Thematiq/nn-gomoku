from typing import Dict, Tuple, Callable

import gymnasium as gym
import numpy as np

from evaluation import *


import torch
def create_check_final_filter() -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([
        create_filter(5, 5, Position.HORIZONTAL),
        create_filter(5, 5, Position.VERTICAL),
        create_filter(5, 5, Position.LEFT_SIDED_DIAGONAL),
        create_filter(5, 5, Position.RIGHT_SIDED_DIAGONAL)
    ]), torch.tensor([[[5.]], [[5.]], [[5.]], [[5.]]])


class GomokuEnv(gym.Env):
    END_REWARD = 100.

    def __init__(self, opponent: Callable, board_size: int, render: bool = False) -> None:
        self._opponent = opponent
        self._board_size = board_size
        self._render = render

        self.evaluator = ConvolutionEvaluation(*create_check_final_filter())
        self.reset()

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1, 1, shape=(self._board_size, self._board_size), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self._board_size ** 2)

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._board = np.zeros((self._board_size, self._board_size), dtype=np.float32)
        self.moves = []

        return self._board, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._board[action // self._board_size, action % self._board_size] = 1
        self.moves.append((action, 1))

        if self.evaluator.evaluate(self._board, None) == np.inf:
            self.render(last=True)
            return self._board, self.END_REWARD, True, False, {'moves': self.moves, 'winner': 'Agent'}

        opponent_action = self._opponent(self._board)
        self._board[opponent_action // self._board_size, opponent_action % self._board_size] = -1
        self.moves.append((opponent_action, -1))

        if (value := self.evaluator.evaluate(self._board, None)) == -np.inf:
            self.render(last=True)
            return self._board, -self.END_REWARD, True, False, {'moves': self.moves, 'winner': 'Opponent'}

        self.render(last=False)
        return self._board, value, False, False, {}

    def render(self, last: bool) -> None:
        if not self._render:
            return

        print(f'Move: {len(self.moves)}')

        last_move = self.moves[-1 if last else -2][0]
        last_move_x, last_move_y = last_move // self._board_size, last_move % self._board_size

        columns = '     ' + ' '.join([chr(ord('A') + i) for i in range(self._board_size)])
        hline = '   ' + '+' + '-' * (2 * self._board_size + 1) + '+'

        print(columns)
        print(hline)

        for i in range(self._board_size - 1, -1, -1):
            symbols = ['.' if cell == 0 else 'X' if cell == 1 else 'O' for cell in self._board[i]]
            row = '{:2d} | {} |'.format(i + 1, ' '.join(symbols))

            if i == last_move_x:
                row = row[:last_move_y * 2 + 6] + ')' + row[last_move_y * 2 + 7:]

            print(row)

        print(hline)
        print(columns)
        print()

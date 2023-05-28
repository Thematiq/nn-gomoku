from typing import Dict, Tuple, Callable

import gymnasium as gym
import numpy as np

from evaluation import *


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

        return self._board, {'moves': self.moves, 'winner': 0}

    def _make_move(self, action: int, player: int) -> float:
        self._board[action // self._board_size, action % self._board_size] = player
        self.moves.append((action, player))

        if self._render:
            self.render(player)

        return self.evaluator.evaluate(self._board, None)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._make_move(action, 1) == np.inf:
            return self._board, self.END_REWARD, True, False, {'moves': self.moves, 'winner': 1}

        opponent_action = self._opponent(self._board)

        if (value := self._make_move(opponent_action, -1)) == -np.inf:
            return self._board, -self.END_REWARD, True, False, {'moves': self.moves, 'winner': -1}

        return self._board, value, False, False, {'moves': self.moves, 'winner': 0}

    def render(self, player: int) -> None:
        print(f'To play: {"agent" if player == 1 else "opponent"}')
        print(f'Move: {len(self.moves)}')

        last_move = self.moves[-1][0]
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

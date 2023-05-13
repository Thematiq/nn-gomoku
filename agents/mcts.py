from __future__ import annotations

import numpy as np

from typing import Tuple, List, Dict, Optional, Callable
from gym_gomoku.envs.util import GomokuUtil

from .agent import Agent

Board = np.ndarray
ActionPos = int
ActionPolicy = Tuple[ActionPos, float]
AvailableActions = List[ActionPolicy]

# Function that decides play style in rollout
RolloutPolicy = Callable[[Board], AvailableActions]

# Function that decides the expansions of tree when we do not have information
ExpandPolicy = Callable[[Board], AvailableActions]


def get_available_positions(board: Board) -> np.ndarray:
    return np.argwhere(board.flatten() == 0)


def default_expand_policy(board: Board) -> AvailableActions:
    positions = get_available_positions(board)
    return zip(positions, np.ones_like(positions) / positions.shape)


def default_rollout_policy(board: Board) -> AvailableActions:
    positions = get_available_positions(board)
    return zip(positions, np.random.uniform(0, 1, size=positions.shape))


class MCTSNode:
    def __init__(self, parent: Optional[MCTSNode], prior: float):
        self._parent = parent
        self._children: Dict[ActionPos, MCTSNode] = {}
        self._visits = 0
        self._q = 0
        # Upper confidence bound
        self._u = 0
        # Prior
        self._p = prior

    @property
    def visits(self) -> int:
        return self._visits

    @property
    def q_val(self) -> float:
        return self._q

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return bool(self._children)

    def expand(self, actions: AvailableActions):
        for action, prob in actions:
            if action not in self._children:
                self._children[action] = MCTSNode(self, prob)

    def select(self, confidence: float) -> Tuple[ActionPos, MCTSNode]:
        return max(self._children.items(),
                   key=lambda x: x[1].eval(confidence))

    def update(self, bp_val):
        self._visits += 1
        self._q = (bp_val - self._q) / self.visits

    def eval(self, confidence):
        self._u = self._p * np.sqrt(self._parent.visits / (1 + self.visits))
        return self._q + confidence * self._u

    def backpropagation(self, bp_val):
        self.update(bp_val)
        if self._parent is not None:
            # swap players
            self._parent.update(-bp_val)


class MCTS:
    def __init__(self, expand_policy: ExpandPolicy, rollout_policy: RolloutPolicy,
                 confidence: float, samples_limit: int, expand_bound: int):
        self._root = MCTSNode(None, 1.0)
        self._expand_policy = expand_policy
        self._rollout_policy = rollout_policy
        self._c = confidence
        self._samples_limit = samples_limit
        self._expand_bound = min(samples_limit, expand_bound)
        self._util = GomokuUtil()

    def __check_terminal(self, board, opponent):
        terminal, winner = self._util.check_five_in_row(board)
        is_winner_opponent = winner == 'white'

        if terminal:
            if is_winner_opponent and opponent:
                return np.inf
            else:
                return -1 * np.inf

        return None

    def __eval_rollout(self, board: Board, opponent) -> float:
        pass

    def __playout(self, board: Board, opponent: bool):
        board = board.copy()
        node = self._root
        while True:
            if node.is_leaf():
                break

            action, node = node.select(self._c)
            board[action] = -1 if opponent else 1
            opponent = not opponent

        actions = self._expand_policy(board)
        is_end, winner = self.__check_terminal(board, opponent)

        if not is_end and node.visits >= self._expand_bound:
            node.expand(actions)

        bp_val = self.__eval_rollout(board)
        node.backpropagation(bp_val)



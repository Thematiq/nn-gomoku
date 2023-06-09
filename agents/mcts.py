from __future__ import annotations

import logging as log
import random

import numpy as np

from tqdm import trange
from typing import Tuple, List, Dict, Optional, Callable
from evaluation import Evaluation
from evaluation.convolution_evaluation import ConvolutionEvaluation
from evaluation.evaluation import MAX_EVALUATION
from evaluation.filters import create_check_final_filter

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
    return np.argwhere(board == 0).flatten()


def create_default_eval():
    return ConvolutionEvaluation(*create_check_final_filter())


def softmax(prob):
    if prob.shape[0] == 0:
        return prob
    prob_norm = prob - np.max(prob)
    prob_exp = np.exp(prob_norm)
    return prob_exp / np.sum(prob_exp)


def default_expand_policy(board: Board) -> AvailableActions:
    positions = get_available_positions(board)
    prob = np.ones(positions.shape)
    return zip(positions, prob)


def default_rollout_policy(board: Board) -> AvailableActions:
    positions = get_available_positions(board)
    return zip(positions, np.ones(positions.shape))


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
    def children(self) -> Dict[ActionPos, MCTSNode]:
        return self._children

    @property
    def visits(self) -> int:
        return self._visits

    @property
    def q_val(self) -> float:
        return self._q

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return not bool(self._children)

    def expand(self, actions: AvailableActions):
        for action, prob in actions:
            if action not in self._children:
                self._children[action] = MCTSNode(self, prob)

    def select(self, confidence: float) -> Tuple[ActionPos, MCTSNode]:
        return random.choice(list(self._children.items()))

    def update(self, bp_val):
        self._q *= self._visits
        self._visits += 1
        self._q = (self._q + bp_val) / self._visits

    def eval(self, confidence):
        return self._q

    def backpropagation(self, bp_val):
        self.update(bp_val)
        if self._parent is not None:
            # swap players
            self._parent.backpropagation(-bp_val)

    def make_root(self):
        self._parent = None

    def __repr__(self):
        return f"Visits: {self.visits} prob: {self.q_val}"


class MCTSAgent(Agent):
    def __init__(self, expand_policy: ExpandPolicy = default_expand_policy,
                 rollout_policy: RolloutPolicy = default_rollout_policy,
                 confidence: float = 1.41, samples_limit: int = 10_000, expand_bound: int = 1,
                 evaluator: Evaluation = create_default_eval(), epsilon: float = 1e-5,
                 board_size: int = 15, rollout_bound: int = 1_000, silent: bool = True):
        self._root = MCTSNode(None, 1.0)
        self._expand_policy = expand_policy
        self._rollout_policy = rollout_policy
        self._c = confidence
        self._board_size = board_size
        self._rollout_bound = rollout_bound
        self._samples_limit = samples_limit
        self._silent_mode = silent
        self._expand_bound = min(samples_limit, expand_bound)
        self._eval = evaluator
        self._epsilon = epsilon

    def __check_terminal(self, board: Board, is_player_opponent: bool) -> float:
        board = board.reshape(self._board_size, self._board_size)
        sign = -1 if is_player_opponent else 1
        evaluation = self._eval.evaluate(board, -1 if is_player_opponent else 1)
        if np.isinf(evaluation):
            if evaluation > 0:
                evaluation = -MAX_EVALUATION
            else:
                evaluation = MAX_EVALUATION
            return sign * evaluation
        return None

    def __eval(self, board: Board, is_player_opponent: bool) -> float:
        board = board.reshape(self._board_size, self._board_size)
        sign = -1 if is_player_opponent else 1
        evaluation = self._eval.evaluate(board, -1 if is_player_opponent else 1)
        if np.isinf(evaluation):
            if evaluation > 0:
                evaluation = -MAX_EVALUATION
            else:
                evaluation = MAX_EVALUATION
        return sign * evaluation

    def __eval_rollout(self, board: Board, opponent: bool) -> float:
        player = opponent

        for _ in range(self._rollout_bound):
            # we pass here a player, so we know the outcome for rollout
            state = self.__check_terminal(board, player)
            if state is not None:
                return state
            action_probs = list(self._rollout_policy(board))
            if len(action_probs) == 0:
                return 0
            next_action = random.choice(action_probs)[0]
            board[next_action] = -1 if opponent else 1
            opponent = not opponent
        return self.__eval(board, player)

    def __playout(self, board: Board, opponent: bool) -> None:
        node = self._root
        while True:
            if node.is_leaf():
                break

            action, node = node.select(self._c)
            board[action] = -1 if opponent else 1
            opponent = not opponent

        actions = self._expand_policy(board)
        state = self.__check_terminal(board, opponent)

        if state is None and node.visits >= self._expand_bound:
            node.expand(actions)

        bp_val = self.__eval_rollout(board, opponent)
        node.backpropagation(bp_val)

    def __play(self, board, opponent) -> ActionPos:
        # Basic heuristic - if map is empty play the center
        if (board == 0).all():
            return board.shape[0] // 2

        it = range(self._samples_limit) if self._silent_mode else trange(self._samples_limit)

        for _ in it:
            board_copy = board.copy()
            self.__playout(board_copy, opponent)
        return max(self._root.children.items(),
                   key=lambda x: x[1].q_val)[0]

    def __update(self, last_move: ActionPos) -> None:
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root.make_root()
        else:
            self._root = MCTSNode(None, 1.0)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
               terminal: bool) -> None:
        self.__update(action)

    def act(self, state: np.ndarray, is_training=True, opponent=False) -> int:
        if opponent:
            self._root = MCTSNode(None, 1.0)
        if not isinstance(state, np.ndarray):
            board = state.board.encode().flatten()
        else:
            board = state.flatten()
        pos = self.__play(board, opponent)
        self.__update(pos)
        return pos

    def opponent_policy(self, state: np.ndarray, *_) -> int:
        return self.act(state, opponent=True)

    def save(self, path: str) -> None:
        pass

    @staticmethod
    def load(path: str) -> 'Agent':
        pass

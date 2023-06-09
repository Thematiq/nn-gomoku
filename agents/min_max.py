import numba
import numpy as np

from agents.agent import Agent
from evaluation.evaluation import Evaluation, MAX_EVALUATION


class AlphaBetaAgent(Agent):
    def __init__(self, depth: int, evaluator: Evaluation):
        self._eval = evaluator
        self._d = depth

    def update(self, state, action, reward, next_state, terminal):
        pass

    def _check_terminal(self, board, opponent):
        sign = 1 if opponent else -1
        evaluation = self._eval.evaluate(board, opponent)
        if np.isinf(evaluation):
            if evaluation > 0:
                evaluation = -MAX_EVALUATION
            else:
                evaluation = MAX_EVALUATION
            return sign * evaluation
        return None

    def _eval_state(self, board, opponent):
        sign = 1 if opponent else -1
        evaluation = self._eval.evaluate(board, opponent)
        if np.isinf(evaluation):
            if evaluation > 0:
                evaluation = -MAX_EVALUATION
            else:
                evaluation = MAX_EVALUATION
        return sign*evaluation

    @numba.jit(forceobj=True)
    def _maximize(self, board, depth, alpha, beta, opponent):
        move_sign = -1 if opponent else 1
        best_val = -1 * np.inf
        best_pos = np.nan
        for current_pos in np.argwhere(board == 0):
            current_pos = tuple(current_pos)

            board[current_pos] = move_sign
            _, current_val = self._alpha_beta(board, depth - 1, alpha, beta, not opponent, False)
            board[current_pos] = 0

            if current_val > best_val:
                best_val = current_val
                best_pos = current_pos

            if current_val > beta:
                break
            alpha = max(alpha, current_val)

        return best_pos, best_val

    @numba.jit(forceobj=True)
    def _minimize(self, board, depth, alpha, beta, opponent):
        move_sign = -1 if opponent else 1
        best_val = np.inf
        best_pos = np.nan
        for current_pos in np.argwhere(board == 0):
            current_pos = tuple(current_pos)

            board[current_pos] = move_sign
            _, current_val = self._alpha_beta(board, depth - 1, alpha, beta, not opponent, True)
            board[current_pos] = 0

            if current_val < best_val:
                best_val = current_val
                best_pos = current_pos

            if current_val < alpha:
                break
            beta = min(beta, current_val)

        return best_pos, best_val

    @numba.jit(forceobj=True)
    def _alpha_beta(self, board, depth, alpha, beta, opponent, maximizing):
        terminal = self._check_terminal(board, opponent)

        if terminal is not None:
            return None, terminal
        if depth == 0:
            return None, self._eval_state(board, opponent)

        if maximizing:
            return self._maximize(board, depth, alpha, beta, opponent)
        else:
            return self._minimize(board, depth, alpha, beta, opponent)

    def act(self, state: np.ndarray, opponent=False, is_training=False):
        if not isinstance(state, np.ndarray):
            board = state.board.encode()
        else:
            board = state
        board = board.astype(np.int32)
        board_size = board.shape[0]

        node, val = self._alpha_beta(board, self._d, -1 * np.inf, np.inf, opponent, False)
        # flatten the result
        return node[1] + (node[0] * board_size)

    def opponent_policy(self, state, *_):
        return self.act(state, opponent=True)

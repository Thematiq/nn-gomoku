import numpy as np

from agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int) -> None:
        np.random.seed(seed)

    def update(self, *_) -> None:
        pass

    def act(self, state: np.ndarray, *_) -> int:
        if isinstance(state, np.ndarray):
            state = state.reshape(-1)
        else:
            state = state.board.encode().reshape(-1)

        zero_indices = np.argwhere(state == 0)
        action = zero_indices[np.random.choice(zero_indices.shape[0])]

        return action.squeeze()

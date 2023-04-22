import numpy as np

from agents.agent import Agent


class RandomAgent(Agent):
    def init(self, seed):
        pass

    def update(self, state, action, reward, next_state, terminal):
        pass

    def act(self, state):
        if isinstance(state, np.ndarray):
            state = state.reshape(-1)
        else:
            state = state.board.encode().reshape(-1)

        zero_indices = np.argwhere(state == 0)
        action = zero_indices[np.random.choice(zero_indices.shape[0])]

        return action.squeeze()

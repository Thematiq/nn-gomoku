from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool) -> None:
        pass

    @abstractmethod
    def act(self, state: np.ndarray, is_training=True) -> int:
        pass

    def save(self, path: str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(path: str) -> 'Agent':
        raise NotImplementedError()

    def opponent_policy(self, state: np.ndarray, *_) -> int:
        return self.act(state)

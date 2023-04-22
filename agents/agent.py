from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def init(self, seed):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, terminal):
        pass

    @abstractmethod
    def act(self, state):
        pass

    def save(self, path):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def opponent_policy(self, state, *_):
        return self.act(state)

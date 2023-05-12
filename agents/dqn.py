from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim

from agents import Agent
from agents.random_agent import RandomAgent
from agents.utils import ExperienceReplay


class QNetwork(torch.nn.Module):
    def __init__(self, board_size: int, hidden_dim: int = 128) -> None:
        super(QNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(board_size ** 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, board_size ** 2)

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQN(Agent):
    r"""
    Double Q-learning agent [1]_ with :math:`\epsilon`-greedy exploration and experience replay buffer. The agent
    uses two Q-networks to stabilize the learning process and avoid overestimation of the Q-values. The main Q-network
    is trained to minimize the Bellman error. The target Q-network is updated with a soft update. This agent follows
    the off-policy learning paradigm and is suitable for environments with discrete action spaces.

    Parameters
    ----------
    board_size : int, default=15
        Size of the Gomoku board.
    seed : int, default=42
        Random seed.
    gamma : float, default=0.99
        Discount factor. `gamma = 0` means no discount, `gamma = 1` means infinite discount.
    learning_rate : float, default=3e-4
        Learning rate of the optimizer.
    epsilon : float, default=0.9
        Initial epsilon-greedy parameter.
    epsilon_decay : float, default=0.999
        Epsilon decay factor. :math:`\epsilon_{t+1} = \epsilon_{t} * \epsilon_{decay}`.
    epsilon_min : float, default=0.01
        Minimum epsilon-greedy parameter.
    soft_update : float, default=0.005
        Soft update factor. `soft_update = 0` means no soft update, `soft_update = 1` means hard update.
    capacity : int, default=10000
        Size of the experience replay buffer.
    batch_size : int, default=64
        Batch size of the samples from the experience replay buffer.
    experience_replay_steps : int, default=5
        Number of experience replay steps per update.

    References
    ----------
    .. [1] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning.
       Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, 2094–2100. Phoenix, Arizona: AAAI Press.
    """

    def __init__(
            self,
            board_size: int,
            seed: int = 42,
            gamma: float = 0.99,
            learning_rate: float = 3e-4,
            epsilon: float = 0.9,
            epsilon_decay: float = 0.999,
            epsilon_min: float = 0.01,
            soft_update: float = 0.005,
            capacity: int = 10000,
            batch_size: int = 64,
            experience_replay_steps: int = 5
    ) -> None:
        torch.manual_seed(seed)

        self.q = QNetwork(board_size)
        self.q_target = QNetwork(board_size)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)

        self.random_agent = RandomAgent(seed)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = ExperienceReplay(capacity, batch_size)
        self.steps = experience_replay_steps

        self.soft_update = soft_update
        self.gamma = gamma

    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, terminal: bool) -> None:
        self.memory.push(state, action, reward, next_state, terminal)

        if len(self.memory) < self.memory.batch_size:
            return

        for _ in range(self.steps):
            states, actions, rewards, next_states, terminals = self.memory.sample()
            q_values = self.q(states).gather(1, actions.type(torch.int64)[..., None])

            with torch.no_grad():
                q_values_target = self.q_target(next_states).max(1)[0]
                target = rewards + (1 - terminals) * self.gamma * q_values_target

            loss = nn.MSELoss()(q_values.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            q, q_target = self.q.state_dict(), self.q_target.state_dict()
            self.q_target.load_state_dict({
                key: deepcopy(q[key]) * self.soft_update + deepcopy(q_target[key]) * (1 - self.soft_update) for key in q
            })

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state: torch.Tensor, is_training=True) -> int:
        if torch.rand(1) < self.epsilon and is_training:
            return self.random_agent.act(state).item()

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)[None, ...]
            logits = self.q(state).squeeze()
            logits[state.flatten() != 0] = -torch.inf

        return logits.argmax().item()

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path))

    def opponent_policy(self, state: torch.Tensor, *_) -> int:
        return self.act(-state, is_training=False)

import random
from collections import deque, namedtuple
from typing import Tuple

import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal'))


class ExperienceReplay:
    def __init__(self, capacity: int, batch_size: int) -> None:
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: Tuple) -> None:
        self.memory.append(Transition(*args))

    def sample(self) -> Tuple:
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        return tuple(map(lambda x: torch.tensor(x, dtype=torch.float32), tuple(batch)))

    def __len__(self) -> int:
        return len(self.memory)

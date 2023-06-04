from collections import deque
import random
from typing import List, Tuple

import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, states: List[BaseData], actions: List[List[int]], rewards: List[torch.Tensor],
            next_states: List[BaseData]):
        for experience_tuple in zip(states, actions, rewards, next_states):
            self.buffer.append(experience_tuple)

    def __len__(self):
        return len(self.buffer)

    def sample(self) -> Tuple[Batch, torch.Tensor, torch.Tensor, Batch]:
        sample_size = self.batch_size if len(self) >= self.batch_size else len(self)
        experience_tuples = random.sample(self.buffer, sample_size)
        return self._batch(experience_tuples)

    @staticmethod
    def _batch(
            experience_tuples: List[Tuple[BaseData, torch.Tensor, torch.Tensor, BaseData]]
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor, Batch]:
        states = Batch.from_data_list([experience_tuple[0] for experience_tuple in experience_tuples])
        actions = torch.cat([experience_tuple[1] for experience_tuple in experience_tuples], dim=0)
        rewards = torch.cat([experience_tuple[2] for experience_tuple in experience_tuples], dim=0)
        next_states = Batch.from_data_list([experience_tuple[3] for experience_tuple in experience_tuples])
        return states, actions, rewards, next_states

from collections import deque
import random
from typing import List, Tuple

import torch
from torch_geometric.data import Data, Batch


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state: Data, actions: List[int], rewards: torch.Tensor, next_state: Data, done: bool):
        self.buffer.append((state, torch.tensor(actions), rewards, next_state, torch.tensor([done])))

    def __len__(self):
        return len(self.buffer)

    def sample(self) -> Tuple[Batch, torch.Tensor, torch.Tensor, Batch, torch.Tensor]:
        sample_size = self.batch_size if len(self) >= self.batch_size else len(self)
        experience_tuples = random.sample(self.buffer, sample_size)
        return self.experience_tuples_to_tensors(experience_tuples)

    @staticmethod
    def experience_tuples_to_tensors(
            experience_tuples: List[Tuple[Data, torch.Tensor, torch.Tensor, Data, torch.Tensor]]
    ) -> tuple[Batch, torch.Tensor, torch.Tensor, Batch, torch.Tensor]:
        states = Batch.from_data_list([experience_tuple[0] for experience_tuple in experience_tuples])
        actions = torch.cat([experience_tuple[1] for experience_tuple in experience_tuples])
        rewards = torch.cat([experience_tuple[2] for experience_tuple in experience_tuples])
        next_states = Batch.from_data_list([experience_tuple[3] for experience_tuple in experience_tuples])
        dones = torch.cat([experience_tuple[4] for experience_tuple in experience_tuples])
        return states, actions, rewards, next_states, dones

from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.nn import aggr


class NetworkHead(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, NetworkHead)
        return obj

    def __init__(self, agent_dim: int, action_dim: int):
        super(NetworkHead, self).__init__()
        self.agent_dim = agent_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(
            self,
            agent_embedding: torch.Tensor,
            action_embedding: torch.Tensor,
            action_index: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        pass


class ActorHead(NetworkHead):

    def __init__(self, *args, **kwargs):
        super(ActorHead, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(self.action_dim, 1)

    def forward(
            self,
            agent_embedding: torch.Tensor,
            action_embedding: torch.Tensor,
            action_index: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        return self.linear(action_embedding).squeeze(), action_index


class CriticHead(NetworkHead):

    def __init__(self, *args, **kwargs):
        super(CriticHead, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(self.agent_dim, 1)

    def forward(
            self,
            agent_embedding: torch.Tensor,
            action_embedding: torch.Tensor,
            action_index: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        return self.linear(agent_embedding).squeeze(), action_index


class DuelingHead(NetworkHead):

    def __init__(self, *args, **kwargs):
        super(DuelingHead, self).__init__(*args, **kwargs)
        self.linear_value = nn.Linear(self.agent_dim, 1)
        self.linear_advantage = nn.Linear(self.action_dim, 1)
        self.mean = aggr.MeanAggregation()

    def forward(
            self,
            agent_embedding: torch.Tensor,
            action_embedding: torch.Tensor,
            action_index: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        state_values = self.linear_value(agent_embedding)[action_index]
        action_advantages = self.linear_advantage(action_embedding)
        action_advantages = action_advantages - self.mean(action_advantages, action_index)[action_index]
        action_values = state_values + action_advantages
        return action_values.squeeze(), action_index

from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn

from src.modules.utils import ResidualStack


class ActorHead(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, ActorHead)
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        pass


class LinearActorHead(ActorHead):

    def __init__(self, input_dim: int, hidden_dim: int, residual_stack_size: int):
        super(LinearActorHead, self).__init__()
        self.res_stack = ResidualStack(input_dim, hidden_dim, residual_stack_size, last_activation=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        res_out = self.res_stack(x)
        return self.linear(res_out).squeeze()

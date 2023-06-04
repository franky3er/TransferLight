from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn

from src.modules.message_passing import NodeAggregation
from src.modules.utils import ResidualStack


class CriticHead(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, CriticHead)
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        pass


class LinearCriticHead(CriticHead):

    def __init__(self, input_dim: int, hidden_dim: int, residual_stack_sizes: Tuple[int, int], aggr_fn: str = "sum"):
        super(LinearCriticHead, self).__init__()
        in_stack_size, out_stack_size = residual_stack_sizes[0], residual_stack_sizes[1]
        self.residual_stack_before_aggr = ResidualStack(
            input_dim, hidden_dim, in_stack_size, last_activation=True) if in_stack_size > 0 else None
        self.aggregation = NodeAggregation(aggr_fn=aggr_fn)
        self.residual_stack_after_aggr = ResidualStack(
            hidden_dim if in_stack_size > 0 else input_dim, hidden_dim, out_stack_size, last_activation=False) \
            if out_stack_size > 0 else None
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        x = self.residual_stack_before_aggr(x) if self.residual_stack_before_aggr is not None else x
        x = self.aggregation(x, index)
        x = self.residual_stack_after_aggr(x) if self.residual_stack_after_aggr else x
        return self.linear(x).squeeze()

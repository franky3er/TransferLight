from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.nn import aggr

from src.modules.utils import ResidualStack


class DQNHead(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, DQNHead)
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor, index: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        pass


class LinearDuelingHead(DQNHead):

    def __init__(self, input_dim: int, hidden_dim: int, value_residual_stack_sizes: Tuple[int, int],
                 advantage_residual_stack_size: int, aggr_fn: str = "sum"):
        super(LinearDuelingHead, self).__init__()
        value_in_stack_size = value_residual_stack_sizes[0]
        value_out_stack_size = value_residual_stack_sizes[1]
        self.value_head_in_res_stack = ResidualStack(input_dim, hidden_dim, value_in_stack_size) \
            if value_in_stack_size > 0 else None
        self.value_head_out_res_stack = ResidualStack(
            hidden_dim if value_in_stack_size > 0 else input_dim, hidden_dim, value_out_stack_size) \
            if value_out_stack_size > 0 else None
        self.value_head_linear = nn.Linear(
            hidden_dim if value_in_stack_size > 0 or value_out_stack_size > 0 else input_dim, 1)

        self.advantage_head_res_stack = ResidualStack(input_dim, hidden_dim, advantage_residual_stack_size) \
            if advantage_residual_stack_size > 0 else None
        self.advantage_head_linear = nn.Linear(hidden_dim if advantage_residual_stack_size > 0 else input_dim, 1)

        if aggr_fn == "sum":
            self.aggr_fn = aggr.SumAggregation()
        elif aggr_fn == "mean":
            self.aggr_fn = aggr.MeanAggregation()

    def forward(self, x: torch.Tensor, index: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        state_values = self._value_stream(x, index)
        action_advantages = self._advantage_stream(x, index)
        action_values = state_values + action_advantages
        return action_values.squeeze(), index

    def _value_stream(self, x: torch.Tensor, index: torch.Tensor):
        x = self.value_head_in_res_stack(x)
        x = aggr.SumAggregation()(x, index)
        x = self.value_head_out_res_stack(x)
        state_values = self.value_head_linear(x)
        return state_values[index]

    def _advantage_stream(self, x: torch.Tensor, index: torch.Tensor):
        x = self.advantage_head_res_stack(x)
        action_advantages = self.advantage_head_linear(x)
        mean_action_advantages = aggr.MeanAggregation()(action_advantages, index)[index]
        return action_advantages - mean_action_advantages
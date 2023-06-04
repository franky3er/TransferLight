from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn

from src.modules.message_passing import NodeAggregation
from src.modules.utils import LinearStack


class ActorHead(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, ActorHead)
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        pass


class FlexibleActorHead(ActorHead):

    def __init__(self, input_dim: int, hidden_dim: int, n_layer: int):
        super(FlexibleActorHead, self).__init__()
        self.layers = LinearStack(input_dim, hidden_dim, 1, n_layer, last_activation=False) if n_layer > 0 else None

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        return self.layers(x).squeeze() if self.layers is not None else x.squeeze()


class CriticHead(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, CriticHead)
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        pass


class FlexibleCriticHead(CriticHead):

    def __init__(self, input_dim: int, hidden_dim: int, n_layer: Tuple[int, int], aggr_fn: str = "sum"):
        super(FlexibleCriticHead, self).__init__()
        n_in_layer, n_out_layer = n_layer[0], n_layer[1]
        self.in_layers = LinearStack(input_dim, hidden_dim, hidden_dim, n_in_layer, last_activation=True) \
            if n_in_layer > 0 else None
        self.aggregation = NodeAggregation(aggr_fn=aggr_fn)
        self.out_layers = LinearStack(
            hidden_dim if n_in_layer > 0 else input_dim, hidden_dim, 1, n_out_layer, last_activation=False) \
            if n_out_layer > 0 else None

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        x = self.in_layers(x) if self.in_layers is not None else x
        x = self.aggregation(x, index)
        return self.out_layers(x) if self.out_layers else x

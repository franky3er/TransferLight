from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn

from src.modules.node_aggregations import NodeAggregation
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


class FixedActorHead(ActorHead):

    def __init__(self, input_dim: int, hidden_dim: int, actions: int, n_layer: int):
        super(FixedActorHead, self).__init__()
        self.layers = self.layers = LinearStack(input_dim, hidden_dim, actions, n_layer, last_activation=False)

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        return self.layers(x)


class FlexibleActorHead(ActorHead):

    def __init__(self, input_dim: int, hidden_dim: int, n_layer: int):
        super(FlexibleActorHead, self).__init__()
        self.layers = LinearStack(input_dim, hidden_dim, 1, n_layer, last_activation=False)

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        return self.layers(x).squeeze()


class CriticHead(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, CriticHead)
        return obj

    @abstractmethod
    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        pass


class FixedCriticHead(CriticHead):

    def __init__(self, input_dim: int, hidden_dim: int, n_layer: int):
        super(FixedCriticHead, self).__init__()
        self.layers = LinearStack(input_dim, hidden_dim, 1, n_layer, last_activation=False)

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        return self.layers(x)


class FlexibleCriticHead(CriticHead):

    def __init__(self, input_dim: int, hidden_dim: int, n_layer: Tuple[int, int], aggregation: Dict):
        super(FlexibleCriticHead, self).__init__()
        n_in_layer, n_out_layer = n_layer[0], n_layer[1]
        self.in_layers = LinearStack(input_dim, hidden_dim, hidden_dim, n_in_layer, last_activation=True)
        self.aggregation = NodeAggregation.create(aggregation["class_name"], aggregation["init_args"])
        self.out_layers = self.layers = LinearStack(input_dim, hidden_dim, 1, n_out_layer, last_activation=False)

    def forward(self, x: torch.Tensor, index: torch.LongTensor):
        x = self.in_layers(x)
        x = self.aggregation(x, index)
        return self.out_layers(x)

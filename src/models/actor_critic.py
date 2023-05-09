from abc import abstractmethod
import sys
from typing import Any, Dict

import torch
from torch import nn
from torch.nn.init import orthogonal_
from torch_geometric.data import HeteroData

from src.models.modules import NodeAggregation, FlexibleCategorical


class ActorNetwork(nn.Module):

    @classmethod
    def create(cls, actor_class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], actor_class_name)(**init_args)
        assert isinstance(obj, ActorNetwork)
        return obj

    @abstractmethod
    def forward(self, state: Any):
        pass

    @abstractmethod
    def full_path(self, state: Any):
        pass

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class CriticNetwork(nn.Module):

    @classmethod
    def create(cls, critic_class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], critic_class_name)(**init_args)
        assert isinstance(obj, CriticNetwork)
        return obj

    @abstractmethod
    def forward(self, state: Any):
        pass

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class ActorCriticNetwork(nn.Module):

    @abstractmethod
    def forward(self, state: Any):
        pass

    @abstractmethod
    def full_pass(self, state: Any):
        pass


class VectorActorNetwork(ActorNetwork):

    def __init__(self, state_size: int, hidden_size: int, action_size: int):
        super(VectorActorNetwork, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(hidden_size, action_size)
        self.apply(self._init_params)

    def forward(self, state: Any):
        x = state.x
        x = self.hidden_layers(x)
        logits = self.output_layer(x)
        return logits

    def full_path(self, state: Any):
        logits = self.forward(state)
        distribution = torch.distributions.categorical.Categorical(logits=logits)
        action = distribution.sample()
        log_prob_action = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob_action, entropy


class VectorCriticNetwork(CriticNetwork):

    def __init__(self, state_size: int, hidden_size: int):
        super(VectorCriticNetwork, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.apply(self._init_params)

    def forward(self, state: Any):
        x = state.x
        x = self.hidden_layers(x)
        value = self.output_layer(x).squeeze()
        return value


class GraphActorNetwork(ActorNetwork):

    def __init__(self, movement_dim: int, phase_dim: int, hidden_dim: int):
        super(GraphActorNetwork, self).__init__()
        self.movement_embedding_layers = nn.Sequential(
            nn.Linear(movement_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.movement_to_phase_aggr = NodeAggregation(aggr_fn="mean")
        self.phase_embedding_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.movement_to_phase_aggr = NodeAggregation(aggr_fn="mean")
        self.apply(self._init_params)
        self.device = "cpu"

    def forward(self, state: HeteroData):
        x_dict = state.x_dict
        edge_index_dict = state.edge_index_dict
        movement_demand = self.movement_embedding_layers(x_dict["movement"])
        logits = self.phase_embedding_layers(
            self.movement_to_phase_aggr(movement_demand, edge_index_dict["movement", "to", "phase"])).squeeze()
        return logits

    def full_path(self, state: HeteroData):
        x_dict = state.x_dict
        edge_index_dict = state.edge_index_dict
        logits = self.forward(state)
        index = edge_index_dict["phase", "to", "intersection"][1]
        distribution = FlexibleCategorical(logits, index).to(self.device)
        actions, action_indices = distribution.sample(return_sample_indices=True)
        log_probs = distribution.log_prob(action_indices)
        entropy = distribution.entropy()
        return actions, log_probs, entropy

    def to(self, device):
        self.device = device
        self.movement_to_phase_aggr.to(device)
        return super(GraphActorNetwork, self).to(device)


class GraphCriticNetwork(CriticNetwork):

    def __init__(self, movement_dim: int, phase_dim: int, intersection_dim: int, hidden_dim: int):
        super(GraphCriticNetwork, self).__init__()
        self.movement_embedding_layers = nn.Sequential(
            nn.Linear(movement_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.phase_embedding_layers = nn.Sequential(
            nn.Linear(phase_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.intersection_embedding_layers = nn.Sequential(
            nn.Linear(intersection_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, state: HeteroData):
        movement_demand = self.movement_embedding_layers(state["movement"])
        print(movement_demand)
        return None

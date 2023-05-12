from abc import abstractmethod
import sys
from typing import Any, Dict, Union, Tuple

import torch
from torch import nn
from torch.nn.init import orthogonal_
from torch_geometric.data import Data, HeteroData

from src.modules.distributions import FlexibleCategorical
from src.modules.traffic_embeddings import TrafficEmbedding
from src.modules.actor_critic_heads import ActorHead, CriticHead


class ActorNetwork(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, ActorNetwork)
        return obj

    def __init__(self, traffic_embedding: Dict, actor_head: Dict):
        super(ActorNetwork, self).__init__()
        self.traffic_embedding = TrafficEmbedding.create(
            traffic_embedding["class_name"], traffic_embedding["init_args"])
        self.actor_head = ActorHead.create(actor_head["class_name"], actor_head["init_args"])

    def forward(self, state: Union[Data, HeteroData]) -> Tuple[torch.Tensor, torch.LongTensor]:
        x, index = self.traffic_embedding(state)
        logits = self.actor_head(x, index)
        return logits, index

    def full_path(self, state: Union[Data, HeteroData]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, index = self.forward(state)
        distribution = FlexibleCategorical(logits, index)
        actions, action_indices = distribution.sample(return_sample_indices=True)
        log_probs = distribution.log_prob(action_indices)
        entropy = distribution.entropy()
        return actions, log_probs, entropy

    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class CriticNetwork(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, CriticNetwork)
        return obj

    def __init__(self, traffic_embedding: Dict, critic_head: Dict):
        super(CriticNetwork, self).__init__()
        self.traffic_embedding = TrafficEmbedding.create(
            traffic_embedding["class_name"], traffic_embedding["init_args"])
        self.critic_head = CriticHead.create(critic_head["class_name"], critic_head["init_args"])

    def forward(self, state: Union[Data, HeteroData]) -> Tuple[torch.Tensor, torch.LongTensor]:
        x, index = self.traffic_embedding(state)
        values = self.critic_head(x, index)
        return values

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

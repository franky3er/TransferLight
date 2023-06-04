import sys
from typing import Dict, Union, Tuple

import torch
from torch import nn
from torch.nn.init import orthogonal_
from torch_geometric.data import Data, HeteroData

from src.modules.distributions import FlexibleCategorical
from src.networks.actor_heads import ActorHead
from src.networks.critic_heads import CriticHead
from src.networks.shared import SharedNetwork


class ActorCritic(nn.Module):

    def __init__(
            self,
            network: Dict,
            actor_head: Dict,
            critic_head: Dict,
            share_network: bool = False,
            learning_rate: float = 0.001,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            entropy_loss_weight: float = 0.01,
            gradient_clipping_max_norm: float = 1.0
    ):
        super(ActorCritic, self).__init__()
        self.shared = share_network
        self.actor_head = ActorHead.create(actor_head["class_name", actor_head["init_args"]])
        self.critic_head = CriticHead.create(critic_head["class_name"], critic_head["init_args"])
        if self.shared:
            self.network = SharedNetwork.create(network["class_name"], network["init_args"])
            self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        else:
            self.network_a = SharedNetwork.create(network["class_name"], network["init_args"])
            self.network_c = SharedNetwork.create(network["class_name"], network["init_args"])
            self.optimizer_a = torch.optim.Adam(list(self.network_a.parameters()) + list(self.actor_head.parameters()),
                                                learning_rate)
            self.optimizer_c = torch.optim.Adam(list(self.network_c.parameters()) + list(self.critic_head.parameters()),
                                                learning_rate)
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.gradient_clipping_max_norm = gradient_clipping_max_norm

    def forward(self, state: HeteroData, actor_out: bool = True, critic_out: bool = True) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self.shared:
            action_embedding, agent_index = self.network(state)
            actor_action_embedding, critic_action_embedding = action_embedding, action_embedding
        else:
            actor_action_embedding, agent_index = self.network_a(state)
            critic_action_embedding, _ = self.network_c(state)
        logits = self.actor_head(actor_action_embedding, agent_index) if actor_out else None
        values = self.critic_head(critic_action_embedding, agent_index) if critic_out else None
        return logits, values, agent_index

    def full_path(self, state: HeteroData) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, index = self.forward(state)
        distribution = FlexibleCategorical(logits, index)
        actions, action_indices = distribution.sample(return_sample_indices=True)
        log_probs = distribution.log_prob(action_indices)
        entropy = distribution.entropy()
        return actions, log_probs, values, entropy

    def critic_path(self, state: HeteroData) -> torch.Tensor:
        values = self.forward(state, actor_out=False)[1]
        return values

    def optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor, entropy_loss: torch.Tensor):
        actor_loss = 0 if actor_loss is None else actor_loss
        critic_loss = 0 if critic_loss is None else critic_loss
        entropy_loss = 0 if entropy_loss is None else entropy_loss
        if self.shared:
            self._shared_optimization_step(actor_loss, critic_loss, entropy_loss)
        else:
            self._unshared_optimization_step(actor_loss, critic_loss, entropy_loss)

    def _shared_optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor,
                                  entropy_loss: torch.Tensor):
        loss = self.actor_loss_weight * actor_loss + self.critic_loss_weight * critic_loss + \
               self.entropy_loss_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.gradient_clipping_max_norm)
        self.optimizer.step()

    def _unshared_optimization_step(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor,
                                    entropy_loss: torch.Tensor):
        actor_loss = self.actor_loss_weight * actor_loss + self.entropy_loss_weight * entropy_loss
        critic_loss = self.critic_loss_weight * critic_loss

        self.optimizer_c.zero_grad()
        self.optimizer_a.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(list(self.network_a.parameters()) + list(self.actor_head.parameters()),
                                      self.gradient_clipping_max_norm)
        torch.nn.utils.clip_grad_norm(list(self.network_c.parameters()) + list(self.critic_head.parameters()),
                                      self.gradient_clipping_max_norm)
        self.optimizer_c.step()
        self.optimizer_a.step()


class ActorNetwork(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, ActorNetwork)
        return obj

    def __init__(self, network: Dict, actor_head: Dict):
        super(ActorNetwork, self).__init__()
        self.traffic_embedding = SharedNetwork.create(
            network["class_name"], network["init_args"])
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

    def __init__(self, network: Dict, critic_head: Dict):
        super(CriticNetwork, self).__init__()
        self.network = SharedNetwork.create(
            network["class_name"], network["init_args"])
        self.critic_head = CriticHead.create(critic_head["class_name"], critic_head["init_args"])

    def forward(self, state: Union[Data, HeteroData]) -> Tuple[torch.Tensor, torch.LongTensor]:
        x, index = self.network(state)
        values = self.critic_head(x, index)
        return values

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

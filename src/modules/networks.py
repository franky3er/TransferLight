from abc import abstractmethod
import sys
from typing import Any, Dict, Tuple

from torch import nn
from torch_geometric.data import HeteroData

from src.rl.problem_formulations import TransferLightProblemFormulation, SimpleTransferLightProblemFormulation
from src.modules.network_modules import TransferLightGraphEmbedding, TransferLightActorCriticHead
from src.modules.utils import group_sum, group_mean


class Network(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, Network)
        return obj


class SimpleTransferLightNetwork(Network):

    def __init__(self, network_type: str, hidden_dim: int = 64, n_residuals: int = 2, n_attention_heads: int = 8,
                 dropout_prob: float = 0.1):
        super(SimpleTransferLightNetwork, self).__init__()
        metadata = SimpleTransferLightProblemFormulation.get_metadata()
        node_dims, edge_dims, pos = metadata["node_dim"], metadata["edge_dim"], metadata["pos"]
        node_updates = [
            ("movement", [("segment", "to_down", "movement"), ("segment", "to_up", "movement")]),
            ("movement", [("movement", "to", "movement")]),
            ("phase", [("movement", "to", "phase")]),
            ("phase", [("phase", "to", "phase")]),
            ("intersection", [("phase", "to", "intersection")])
        ]
        self.embed = TransferLightGraphEmbedding(node_dims, edge_dims, pos, hidden_dim, node_updates,
                                                 n_attention_heads, n_residuals, dropout_prob)

        self.network_type = network_type
        if network_type == "QNet":
            self.linear_state_value = nn.Linear(node_dims["movement"], 1)
            self.linear_action_advantage = nn.Linear(node_dims["phase"], 1)
        elif network_type == "ActorCriticNet":
            self.head = TransferLightActorCriticHead(node_dims, edge_dims, hidden_dim)
            #self.linear_actor = nn.Linear(node_dims["phase"], 1)
            #self.linear_critic = nn.Linear(node_dims["movement"], 1)
        else:
            raise Exception(f'network_type "{network_type}" not implemented')

    def forward(self, state: HeteroData) -> HeteroData:
        state = state.clone()
        embedded_state = self.embed(state)
        return self.head(embedded_state)


class TransferLightNetwork(Network):

    def __init__(self, network_type: str, hidden_dim: int = 64, n_residuals: int = 2, n_attention_heads: int = 8,
                 dropout_prob: float = 0.1):
        super(TransferLightNetwork, self).__init__()
        metadata = TransferLightProblemFormulation.get_metadata()
        node_dims, edge_dims, pos = metadata["node_dim"], metadata["edge_dim"], metadata["pos"]
        node_updates = [
            ("segment", [("segment", "to_up", "segment"), ("segment", "to_down", "segment")]),
            ("lane", [("segment", "to", "lane")]),
            ("movement", [("lane", "to_down", "movement"), ("lane", "to_up", "movement")]),
            #("movement", [("movement", "to_down", "movement"), ("movement", "to_up", "movement")]),
            ("phase", [("movement", "to", "phase")]),
            ("phase", [("phase", "to", "phase")]),
            ("intersection", [("phase", "to", "intersection")])
        ]
        self.embed = TransferLightGraphEmbedding(node_dims, edge_dims, pos, hidden_dim, node_updates,
                                                 n_attention_heads, n_residuals, dropout_prob)

        self.network_type = network_type
        if network_type == "QNet":
            self.linear_state_value = nn.Linear(node_dims["movement"], 1)
            self.linear_action_advantage = nn.Linear(node_dims["phase"], 1)
        elif network_type == "ActorCriticNet":
            self.head = TransferLightActorCriticHead(node_dims, edge_dims, hidden_dim)
            #self.linear_actor = nn.Linear(node_dims["phase"], 1)
            #self.linear_critic = nn.Linear(node_dims["movement"], 1)
        else:
            raise Exception(f'network_type "{network_type}" not implemented')

    def forward(self, state: HeteroData) -> HeteroData:
        state = state.clone()
        embedded_state = self.embed(state)
        return self.head(embedded_state)

    def dueling_dqn_head(self, embedded_state: HeteroData):
        edge_index_movement_to_intersection = embedded_state["movement", "to", "intersection"].edge_index
        movement_embedding = embedded_state["movement"].x[edge_index_movement_to_intersection[0]]
        phase_embedding = embedded_state["phase"].x
        action_index = embedded_state["phase", "to", "intersection"].edge_index[1]
        state_values = group_sum(self.linear_state_value(movement_embedding), edge_index_movement_to_intersection[1])
        state_values = state_values[action_index]
        action_advantages = self.linear_state_value(phase_embedding).squeeze()
        action_advantages = action_advantages - group_mean(action_advantages, action_index)[action_index]
        action_values = state_values + action_advantages
        return action_values.squeeze(), action_index

    def actor_critic_head(self, embedded_state: HeteroData):
        edge_index_movement_to_intersection = embedded_state["movement", "to", "intersection"].edge_index
        movement_embedding = embedded_state["movement"].x[edge_index_movement_to_intersection[0]]
        phase_embedding = embedded_state["phase"].x
        state_value = group_sum(self.linear_critic(movement_embedding),
                                group_index=edge_index_movement_to_intersection[1])
        action_index = embedded_state["phase", "to", "intersection"].edge_index[1]
        action_logits = self.linear_actor(phase_embedding)
        return state_value.squeeze(), action_logits.squeeze(), action_index

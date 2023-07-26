from abc import abstractmethod
import sys
from typing import Any, Dict, Tuple

from torch import nn
from torch_geometric.data import HeteroData

from src.rl.problem_formulations import GeneraLightProblemFormulation
from src.modules.network_modules import (GeneraLightLaneDemandEmbedding,
                                         GeneraLightMovementDemandEmbedding,
                                         GeneraLightPhaseDemandEmbedding,
                                         GeneraLightIntersectionDemandEmbedding)
from src.modules.utils import group_sum, group_mean


class NetworkBody(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, NetworkBody)
        return obj

    def forward(self, state: HeteroData) -> Any:
        embedded_state = self.embed(state)
        return self.head(embedded_state)

    @abstractmethod
    def embed(self, state: HeteroData) -> HeteroData:
        pass

    @abstractmethod
    def head(self, embedded_state: HeteroData) -> Any:
        pass


class GeneraLightNetwork(NetworkBody):

    def __init__(self, network_type: str, hidden_dim: int = 64, n_residuals: int = 2, n_attention_heads: int = 8,
                 dropout_prob: float = 0.1):
        super(GeneraLightNetwork, self).__init__()
        self.network_type = network_type
        if network_type == "DuelingDQN":
            self.linear_state_value = nn.Linear(hidden_dim, 1)
            self.linear_action_advantage = nn.Linear(hidden_dim, 1)
        elif network_type == "ActorCritic":
            self.linear_actor = nn.Linear(hidden_dim, 1)
            self.linear_critic = nn.Linear(hidden_dim, 1)
        else:
            raise Exception(f'network_type "{network_type}" not implemented')
        metadata = GeneraLightProblemFormulation.get_metadata()
        node_dim, edge_dim = metadata["node_dim"], metadata["edge_dim"]
        self.node_dim, self.edge_dim = node_dim, edge_dim
        self.lane_demand_embedding = GeneraLightLaneDemandEmbedding(
            lane_segment_dim=node_dim["lane_segment"],
            lane_dim=node_dim["lane"],
            lane_segment_to_lane_edge_dim=edge_dim[("lane_segment", "to", "lane")],
            output_dim=hidden_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals,
            dropout_prob=dropout_prob
        )
        self.movement_demand_embedding = GeneraLightMovementDemandEmbedding(
            lane_dim=hidden_dim,
            movement_dim=node_dim["movement"],
            lane_to_downstream_movement_edge_dim=edge_dim[("lane", "to_downstream", "movement")],
            lane_to_upstream_movement_edge_dim=edge_dim[("lane", "to_upstream", "movement")],
            movement_to_movement_edge_dim=edge_dim[("movement", "to", "movement")],
            movement_to_movement_hops=2,
            output_dim=hidden_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals,
            dropout_prob=dropout_prob
        )
        self.phase_demand_embedding = GeneraLightPhaseDemandEmbedding(
            movement_dim=hidden_dim,
            phase_dim=node_dim["phase"],
            movement_to_phase_edge_dim=edge_dim[("movement", "to", "phase")],
            phase_to_phase_edge_dim=edge_dim[("phase", "to", "phase")],
            output_dim=hidden_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals,
            dropout_prob=dropout_prob
        )
        self.intersection_demand_embedding = GeneraLightIntersectionDemandEmbedding(
            movement_dim=hidden_dim,
            intersection_dim=node_dim["intersection"],
            movement_to_intersection_edge_dim=edge_dim[("movement", "to", "intersection")],
            output_dim=hidden_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals,
            dropout_prob=dropout_prob
        )

    def embed(self, state: HeteroData) -> HeteroData:
        state = state.clone()
        state["lane"].x = self.lane_demand_embedding(
            state["lane_segment"].x if self.node_dim["lane_segment"] > 0 else None,
            state["lane"].x if self.node_dim["lane"] > 0 else None,
            state["lane_segment", "to", "lane"].edge_attr
            if self.edge_dim[("lane_segment", "to", "lane")] > 0 else None,
            state["lane_segment", "to", "lane"].edge_index
        )
        state["movement"].x = self.movement_demand_embedding(
            state["lane"].x,
            state["movement"].x if self.node_dim["movement"] > 0 else None,
            state["lane", "to_downstream", "movement"].edge_attr
            if self.edge_dim[("lane", "to_downstream", "movement")] > 0 else None,
            state["lane", "to_upstream", "movement"].edge_attr
            if self.edge_dim[("lane", "to_upstream", "movement")] > 0 else None,
            state["movement", "to", "movement"].edge_attr
            if self.edge_dim[("movement", "to", "movement")] > 0 else None,
            state["lane", "to_downstream", "movement"].edge_index,
            state["lane", "to_upstream", "movement"].edge_index,
            state["movement", "to", "movement"].edge_index
        )
        state["phase"].x = self.phase_demand_embedding(
            state["movement"].x,
            state["phase"].x if self.node_dim["phase"] > 0 else None,
            state["movement", "to", "phase"].edge_attr if self.edge_dim[("movement", "to", "phase")] > 0 else None,
            state["phase", "to", "phase"].edge_attr if self.edge_dim[("phase", "to", "phase")] > 0 else None,
            state["movement", "to", "phase"].edge_index,
            state["phase", "to", "phase"].edge_index
        )
        return state

    def head(self, embedded_state: HeteroData) -> Any:
        if self.network_type == "DuelingDQN":
            return self.dueling_dqn_head(embedded_state)
        elif self.network_type == "ActorCritic":
            return self.actor_critic_head(embedded_state)

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

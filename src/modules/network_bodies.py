from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj

from src.rl.problem_formulations import GeneraLightProblemFormulation
from src.modules.network_modules import (GeneraLightLaneDemandEmbedding,
                                         GeneraLightMovementDemandEmbedding,
                                         GeneraLightPhaseDemandEmbedding,
                                         GeneraLightIntersectionDemandEmbedding)


class NetworkBody(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, NetworkBody)
        return obj

    @abstractmethod
    def forward(self, state: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        pass


class GeneraLightNetwork(NetworkBody):

    def __init__(self, output_dim: int = 64, n_residuals: int = 2, n_attention_heads: int = 8):
        super(GeneraLightNetwork, self).__init__()
        metadata = GeneraLightProblemFormulation.get_metadata()
        node_dim, edge_dim = metadata["node_dim"], metadata["edge_dim"]
        self.node_dim, self.edge_dim = node_dim, edge_dim
        self.lane_demand_embedding = GeneraLightLaneDemandEmbedding(
            lane_segment_dim=node_dim["lane_segment"],
            lane_dim=node_dim["lane"],
            lane_segment_to_lane_edge_dim=edge_dim[("lane_segment", "to", "lane")],
            output_dim=output_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals
        )
        self.movement_demand_embedding = GeneraLightMovementDemandEmbedding(
            lane_dim=output_dim,
            movement_dim=node_dim["movement"],
            lane_to_downstream_movement_edge_dim=edge_dim[("lane", "to_downstream", "movement")],
            lane_to_upstream_movement_edge_dim=edge_dim[("lane", "to_upstream", "movement")],
            movement_to_movement_edge_dim=edge_dim[("movement", "to", "movement")],
            movement_to_movement_hops=2,
            output_dim=output_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals
        )
        self.phase_demand_embedding = GeneraLightPhaseDemandEmbedding(
            movement_dim=output_dim,
            phase_dim=node_dim["phase"],
            movement_to_phase_edge_dim=edge_dim[("movement", "to", "phase")],
            phase_to_phase_edge_dim=edge_dim[("phase", "to", "phase")],
            output_dim=output_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals
        )
        self.intersection_demand_embedding = GeneraLightIntersectionDemandEmbedding(
            movement_dim=output_dim,
            intersection_dim=node_dim["intersection"],
            movement_to_intersection_edge_dim=edge_dim[("movement", "to", "intersection")],
            output_dim=output_dim,
            heads=n_attention_heads,
            n_residuals=n_residuals
        )

    def forward(self, state: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        lane_embedding = self.lane_demand_embedding(
            state["lane_segment"].x if self.node_dim["lane_segment"] > 0 else None,
            state["lane"].x if self.node_dim["lane"] > 0 else None,
            state["lane_segment", "to", "lane"].edge_attr
            if self.edge_dim[("lane_segment", "to", "lane")] > 0 else None,
            state["lane_segment", "to", "lane"].edge_index
        )
        movement_embedding = self.movement_demand_embedding(
            lane_embedding,
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
        phase_embedding = self.phase_demand_embedding(
            movement_embedding,
            state["phase"].x if self.node_dim["phase"] > 0 else None,
            state["movement", "to", "phase"].edge_attr if self.edge_dim[("movement", "to", "phase")] > 0 else None,
            state["phase", "to", "phase"].edge_attr if self.edge_dim[("phase", "to", "phase")] > 0 else None,
            state["movement", "to", "phase"].edge_index,
            state["phase", "to", "phase"].edge_index
        )
        intersection_embedding = self.intersection_demand_embedding(
            movement_embedding,
            state["intersection"].x if self.node_dim["intersection"] > 0 else None,
            state["movement", "to", "intersection"].edge_attr
            if self.edge_dim[("movement", "to", "intersection")] > 0 else None,
            state["movement", "to", "intersection"].edge_index
        )
        agent_embedding, action_embedding = intersection_embedding, phase_embedding
        index = state["phase", "to", "intersection"].edge_index[1]
        return agent_embedding, action_embedding, index

    def _movement_demand_embedding(
            self,
            lane_segment_x: torch.Tensor,
            movement_x: torch.Tensor,
            lane_segment_to_downstream_movement_edge_attr: torch.Tensor,
            lane_segment_to_upstream_movement_edge_attr: torch.Tensor,
            lane_segment_to_downstream_movement_edge_index: Adj,
            lane_segment_to_upstream_movement_edge_index: Adj
    ) -> torch.Tensor:
        downstream_embedding = self.lane_segment_to_downstream_movement(
            lane_segment_x, movement_x, lane_segment_to_downstream_movement_edge_attr,
            lane_segment_to_downstream_movement_edge_index
        )
        upstream_embedding = self.lane_segment_to_upstream_movement(
            lane_segment_x, movement_x, lane_segment_to_upstream_movement_edge_attr,
            lane_segment_to_upstream_movement_edge_index
        )
        movement_embedding = self.lane_segment_to_movement_linear_stack(
            torch.cat([downstream_embedding, upstream_embedding], dim=1)
        )
        return movement_embedding

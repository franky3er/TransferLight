from abc import abstractmethod
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj

from src.modules.message_passing import (NodeAggregation,
                                         AttentionDirectedBipartiteMessagePassing,
                                         GeneraLightLaneDemandEmbedding,
                                         GeneraLightMovementDemandEmbedding,
                                         GeneraLightPhaseDemandEmbedding,
                                         FRAPPhaseDemandEmbedding,
                                         FRAPPhasePairEmbedding,
                                         FRAPPhaseCompetition)


class SharedNetwork(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, SharedNetwork)
        return obj

    @abstractmethod
    def forward(self, state: HeteroData) -> Tuple[torch.Tensor, torch.LongTensor]:
        pass


class GeneraLightNetwork(SharedNetwork):

    def __init__(self, hidden_dim: int):
        super(GeneraLightNetwork, self).__init__()
        self.lane_demand_embedding = GeneraLightLaneDemandEmbedding(
            lane_segment_dim=2, lane_dim=0, lane_segment_to_lane_edge_dim=1,
            output_dim=hidden_dim, heads=8, n_layer_message=2, n_layer_update=2
        )
        self.movement_demand_embedding = GeneraLightMovementDemandEmbedding(
            lane_dim=hidden_dim, movement_dim=3, lane_to_downstream_movement_edge_dim=1,
            lane_to_upstream_movement_edge_dim=1, output_dim=hidden_dim, heads=8,
            n_layer_update=2, n_layer_message=2, n_layer_output=2
        )
        self.phase_demand_embedding = GeneraLightPhaseDemandEmbedding(
            movement_dim=hidden_dim, phase_dim=1, movement_to_phase_edge_dim=2, phase_to_phase_edge_dim=1,
            output_dim=hidden_dim, heads=8, n_layer_message=2, n_layer_update=2
        )
        self.phase_pair_embedding = FRAPPhasePairEmbedding(phase_dim=hidden_dim)
        self.phase_competition = FRAPPhaseCompetition(phase_dim=hidden_dim, n_layer=3)

    def forward(self, state: HeteroData) -> Tuple[torch.Tensor, torch.LongTensor]:
        lane_demand_embedding = self.lane_demand_embedding(
            state["lane_segment"].x, None, state["lane_segment", "to", "lane"].edge_attr,
            state["lane_segment", "to", "lane"].edge_index
        )
        movement_demand_embedding = self.movement_demand_embedding(
            lane_demand_embedding, state["movement"].x, state["lane", "to_downstream", "movement"].edge_attr,
            state["lane", "to_upstream", "movement"].edge_attr, state["lane", "to_downstream", "movement"].edge_index,
            state["lane", "to_upstream", "movement"].edge_index
        )
        phase_demand_embedding = self.phase_demand_embedding(
            movement_demand_embedding, state["phase"].x, state["movement", "to", "phase"].edge_attr,
            state["phase", "to", "phase"].edge_attr, state["movement", "to", "phase"].edge_index,
            state["phase", "to", "phase"].edge_index
        )
        index = state["phase", "to", "intersection"].edge_index[1]
        return phase_demand_embedding, index

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

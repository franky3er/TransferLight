from abc import abstractmethod
import sys
from typing import Dict, Tuple, Union

import torch
from torch import nn
from torch_geometric import nn as pyg_nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import NodeType, Adj

from src.modules.utils import LinearStack
from src.modules.message_passing import (NodeAggregation,
                                         AttentionDirectedBipartiteMessagePassing,
                                         GeneraLightLaneDemandEmbedding,
                                         GeneraLightMovementDemandEmbedding,
                                         GeneraLightPhaseDemandEmbedding,
                                         FRAPPhaseDemandEmbedding,
                                         FRAPPhasePairEmbedding,
                                         FRAPPhaseCompetition)


class TrafficEmbedding(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, TrafficEmbedding)
        return obj

    def __init__(self):
        super(TrafficEmbedding, self).__init__()

    @abstractmethod
    def forward(self, state: Union[Data, HeteroData]) -> Tuple[torch.Tensor, torch.LongTensor]:
        pass


class GeneraLightTrafficEmbedding(TrafficEmbedding):

    def __init__(self, hidden_dim: int):
        super(GeneraLightTrafficEmbedding, self).__init__()
        self.lane_demand_embedding = GeneraLightLaneDemandEmbedding(
            lane_segment_dim=2, lane_dim=0, lane_segment_to_lane_edge_dim=1, hidden_dim=hidden_dim,
            output_dim=hidden_dim, heads=4, n_layer_message=2, n_layer_update=2
        )
        self.movement_demand_embedding = GeneraLightMovementDemandEmbedding(
            lane_dim=hidden_dim, movement_dim=3, lane_to_downstream_movement_edge_dim=1,
            lane_to_upstream_movement_edge_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim, heads=4,
            n_layer_update=2, n_layer_message=2, n_layer_output=2
        )
        self.phase_demand_embedding = GeneraLightPhaseDemandEmbedding(
            movement_dim=hidden_dim, phase_dim=1, movement_to_phase_edge_dim=2, phase_to_phase_edge_dim=1,
            hidden_dim=hidden_dim, output_dim=hidden_dim, heads=4, n_layer_message=2, n_layer_update=2
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


class HomogeneousTrafficEmbedding(TrafficEmbedding):

    def __init__(self, intersection_embedding: Dict):
        super(HomogeneousTrafficEmbedding, self).__init__()
        self.intersection_embedding = IntersectionEmbedding.create(
            intersection_embedding["class_name"], intersection_embedding["init_args"])

    def forward(self, state: Data) -> Tuple[torch.Tensor, torch.LongTensor]:
        x = state.x
        edge_index = state.edge_index
        x = self.intersection_embedding(x, edge_index)
        return x, None


class HeterogeneousTrafficEmbedding(TrafficEmbedding):

    def __init__(self,
                 movement_embedding: Dict,
                 movement_to_phase_aggregation: Dict,
                 phase_embedding: Dict):
        super(HeterogeneousTrafficEmbedding, self).__init__()
        self.movement_embedding = MovementEmbedding.create(
            movement_embedding["class_name"], movement_embedding["init_args"])
        self.movement_to_phase_aggregation = MovementToPhaseAggregation.create(
            movement_to_phase_aggregation["class_name"], movement_to_phase_aggregation["init_args"])
        self.phase_embedding = PhaseEmbedding.create(
            phase_embedding["class_name"], phase_embedding["init_args"])

    def forward(self, state: HeteroData) -> Tuple[torch.Tensor, torch.LongTensor]:
        movement_embedding = self.movement_embedding(state["movement"].x,
                                                     None)
        phase_embedding = self.movement_to_phase_aggregation(movement_embedding,
                                                             state["movement", "to", "phase"].edge_index)
        x = self.phase_embedding(phase_embedding, state["phase", "to", "phase"].edge_index)
        index = state["phase", "to", "intersection"].edge_index[1]
        return x, index


class MovementEmbedding(pyg_nn.MessagePassing):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, MovementEmbedding)
        return obj

    def __init__(self):
        super(MovementEmbedding, self).__init__()

    @abstractmethod
    def forward(self, x: NodeType, edge_index: Adj):
        pass


class LinearMovementEmbedding(MovementEmbedding):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layer: int):
        super(LinearMovementEmbedding, self).__init__()
        self.layers = LinearStack(input_dim, hidden_dim, output_dim, n_layer, last_activation=True)

    def forward(self, x: NodeType, edge_index: Adj):
        return self.layers(x)


class MovementToPhaseAggregation(pyg_nn.MessagePassing):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, MovementToPhaseAggregation)
        return obj

    @abstractmethod
    def forward(self, x: NodeType, edge_index: Adj):
        pass


class SimpleMovementToPhaseAggregation(MovementToPhaseAggregation):

    def __init__(self, aggr_fn: str):
        super(SimpleMovementToPhaseAggregation, self).__init__(aggr=aggr_fn)

    def forward(self, x: NodeType, edge_index: Adj):
        x, edge_index, offset = self._update_x_and_edge_index(x, edge_index)
        x = self.propagate(edge_index, x=x)
        return x[offset:]

    def _update_x_and_edge_index(self, x_src: torch.Tensor, edge_index: Adj) -> Tuple[torch.Tensor, Adj, int]:
        device = x_src.get_device()
        offset = x_src.size(0)
        n_src_nodes = x_src.size(0)
        n_dest_nodes = torch.max(edge_index[1]) + 1
        dim_src_nodes = x_src.size(1)
        x_dest = torch.zeros(n_dest_nodes, dim_src_nodes, device=device)
        x = torch.cat([x_src, x_dest], dim=0)
        new_edge_index = edge_index.clone()
        new_edge_index[1] += n_src_nodes
        return x, new_edge_index, offset

    def to(self, device):
        self.device = device
        return super(SimpleMovementToPhaseAggregation, self).to(device)


class PhaseEmbedding(pyg_nn.MessagePassing):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, PhaseEmbedding)
        return obj

    @abstractmethod
    def forward(self, x: NodeType, edge_index: Adj):
        pass


class LinearPhaseEmbedding(PhaseEmbedding):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layer: int):
        super(LinearPhaseEmbedding, self).__init__()
        self.layers = LinearStack(input_dim, hidden_dim, output_dim, n_layer, last_activation=True)

    def forward(self, x: NodeType, edge_index: Adj):
        return self.layers(x)


class CompetitionPhaseEmbedding(PhaseEmbedding):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(CompetitionPhaseEmbedding, self).__init__(aggr="mean")
        self.linear_layer_neighbor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_layer_target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_layer_out = nn.Sequential(
            nn.Linear(2*hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: NodeType, edge_index: Adj):
        neighbor = self.linear_layer_neighbor(x)
        target = self.linear_layer_target(x)
        return self.propagate(edge_index, target=target, neighbor=neighbor)

    def message(self, neighbor_j: torch.Tensor) -> torch.Tensor:
        return neighbor_j

    def update(self, message: torch.Tensor, target: torch.Tensor):
        concatenated = torch.cat([target, message], dim=1)
        out = self.linear_layer_out(concatenated)
        return out


class IntersectionEmbedding(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, IntersectionEmbedding)
        return obj

    @abstractmethod
    def forward(self, x: NodeType, edge_index: Adj):
        pass


class LinearIntersectionEmbedding(IntersectionEmbedding):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim, n_layer: int):
        super(LinearIntersectionEmbedding, self).__init__()
        self.layers = LinearStack(input_dim, hidden_dim, output_dim, n_layer, last_activation=True)

    def forward(self, x: NodeType, edge_index: Adj):
        return self.layers(x)

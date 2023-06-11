import torch
from torch import nn
from torch_geometric.typing import Adj

from src.modules.message_passing import HomoNeighborhoodAttention, HeteroNeighborhoodAttention
from src.modules.base_modules import ResidualStack


class GeneraLightLaneDemandEmbedding(nn.Module):

    def __init__(
            self,
            lane_segment_dim: int,
            lane_dim: int,
            lane_segment_to_lane_edge_dim: int,
            output_dim: int,
            heads: int,
            n_residuals: int
    ):
        super(GeneraLightLaneDemandEmbedding, self).__init__()
        self.lane_demand_embedding = HeteroNeighborhoodAttention(
            lane_segment_dim, lane_dim, lane_segment_to_lane_edge_dim, output_dim, heads, n_residuals,
            positional_encoding_method="alibi")

    def forward(
            self,
            lane_segment_x: torch.Tensor,
            lane_x: torch.Tensor,
            lane_segment_to_lane_edge_attr: torch.Tensor,
            lane_segment_to_lane_edge_index: Adj
    ) -> torch.Tensor:
        return self.lane_demand_embedding(
            lane_segment_x, lane_x, lane_segment_to_lane_edge_attr, lane_segment_to_lane_edge_index)


class GeneraLightMovementDemandEmbedding(nn.Module):

    def __init__(
            self,
            lane_dim: int,
            movement_dim: int,
            lane_to_downstream_movement_edge_dim: int,
            lane_to_upstream_movement_edge_dim: int,
            movement_to_movement_edge_dim: int,
            movement_to_movement_hops: int,
            output_dim: int,
            heads: int,
            n_residuals: int
    ):
        super(GeneraLightMovementDemandEmbedding, self).__init__()
        self.incoming_approach_embedding = HeteroNeighborhoodAttention(
            lane_dim, movement_dim, lane_to_downstream_movement_edge_dim, output_dim, heads, n_residuals,
            positional_encoding_method="alibi")
        self.outgoing_approach_embedding = HeteroNeighborhoodAttention(
            lane_dim, movement_dim, lane_to_upstream_movement_edge_dim, output_dim, heads, n_residuals,
            positional_encoding_method="alibi")
        self.combined_movement_embedding = ResidualStack(2 * output_dim, output_dim, n_residuals,
                                                         last_activation=True)
        self.movement_to_movement_embedding = nn.ModuleList()
        for _ in range(movement_to_movement_hops):
            self.movement_to_movement_embedding.append(
                HomoNeighborhoodAttention(output_dim, movement_to_movement_edge_dim, output_dim, heads, n_residuals)
            )

    def forward(
            self,
            lane_x: torch.Tensor,
            movement_x: torch.Tensor,
            lane_to_downstream_movement_edge_attr: torch.Tensor,
            lane_to_upstream_movement_edge_attr: torch.Tensor,
            movement_to_movement_edge_attr: torch.Tensor,
            lane_to_downstream_movement_edge_index: Adj,
            lane_to_upstream_movement_edge_index: Adj,
            movement_to_movement_edge_index: Adj,
    ):
        incoming_approach_embedding = self.incoming_approach_embedding(
            lane_x, movement_x, lane_to_downstream_movement_edge_attr, lane_to_downstream_movement_edge_index)
        outgoing_approach_embedding = self.outgoing_approach_embedding(
            lane_x, movement_x, lane_to_upstream_movement_edge_attr, lane_to_upstream_movement_edge_index)
        movement_embedding = self.combined_movement_embedding(
            torch.cat([incoming_approach_embedding, outgoing_approach_embedding], dim=-1))
        for movement_to_movement_embedding in self.movement_to_movement_embedding:
            movement_embedding = movement_to_movement_embedding(
                movement_embedding, movement_to_movement_edge_attr, movement_to_movement_edge_index)
        return movement_embedding


class GeneraLightPhaseDemandEmbedding(nn.Module):

    def __init__(
            self,
            movement_dim: int,
            phase_dim: int,
            movement_to_phase_edge_dim: int,
            phase_to_phase_edge_dim: int,
            output_dim: int,
            heads: int,
            n_residuals: int
    ):
        super(GeneraLightPhaseDemandEmbedding, self).__init__()
        self.phase_demand_embedding = HeteroNeighborhoodAttention(
            movement_dim, phase_dim, movement_to_phase_edge_dim, output_dim, heads, n_residuals,
            positional_encoding_method=None)
        self.phase_competition = HomoNeighborhoodAttention(
            output_dim, phase_to_phase_edge_dim, output_dim, heads, n_residuals
        )

    def forward(self, movement_x: torch.Tensor, phase_x: torch.Tensor, movement_to_phase_edge_attr: torch.Tensor,
                phase_to_phase_edge_attr: torch.Tensor, movement_to_phase_edge_index: Adj,
                phase_to_phase_edge_index: Adj):
        phase_demand = self.phase_demand_embedding(movement_x, phase_x, movement_to_phase_edge_attr,
                                                   movement_to_phase_edge_index)
        return self.phase_competition(phase_demand, phase_to_phase_edge_attr, phase_to_phase_edge_index)


class GeneraLightIntersectionDemandEmbedding(nn.Module):

    def __init__(
            self,
            movement_dim: int,
            intersection_dim: int,
            movement_to_intersection_edge_dim: int,
            output_dim: int,
            heads: int,
            n_residuals: int
    ):
        super(GeneraLightIntersectionDemandEmbedding, self).__init__()
        self.intersection_demand_embedding = HeteroNeighborhoodAttention(
            movement_dim, intersection_dim, movement_to_intersection_edge_dim, output_dim, heads, n_residuals
        )

    def forward(
            self,
            movement_x: torch.Tensor,
            intersection_x: torch.Tensor,
            movement_to_intersection_edge_attr: torch.Tensor,
            movement_to_intersection_edge_index: torch.Tensor
    ):
        return self.intersection_demand_embedding(movement_x, intersection_x, movement_to_intersection_edge_attr,
                                                  movement_to_intersection_edge_index)

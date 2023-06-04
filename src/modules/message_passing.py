from abc import abstractmethod
from typing import Any, Tuple

import torch
from torch import nn
from torch_geometric.nn import aggr
from torch_geometric.typing import Adj
from torch_geometric.utils import softmax

from src.modules.utils import LinearStack, ResidualStack


class DirectedBipartiteMessagePassing(nn.Module):

    def __init__(self, aggr_fn: str = "sum"):
        super(DirectedBipartiteMessagePassing, self).__init__()
        self.aggr_fn = None
        if aggr_fn == "sum":
            self.aggr_fn = aggr.SumAggregation()
        elif aggr_fn == "mean":
            self.aggr_fn = aggr.MeanAggregation()

    @abstractmethod
    def forward(self, *args: Any) -> torch.Tensor:
        pass

    def propagate(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, edge_index: Adj)\
            -> torch.Tensor:
        x_src_expanded, x_dst_expanded = None, None
        if x_src is not None:
            x_src_expanded = x_src[edge_index[0]]
        if x_dst is not None:
            x_dst_expanded = x_dst[edge_index[1]]
        index = edge_index[1]
        messages = self.message(x_src_expanded, x_dst_expanded, edge_attr, index)
        aggr_messages = self.aggregate(messages, index)
        return self.update(x_dst, aggr_messages)

    @abstractmethod
    def message(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor)\
            -> torch.Tensor:
        pass

    def aggregate(self, messages: torch.Tensor, index: torch.LongTensor):
        return self.aggr_fn(messages, index)

    @abstractmethod
    def update(self, x_dst: torch.Tensor, aggr_messages: torch.Tensor) -> torch.Tensor:
        pass


class NodeAggregation(DirectedBipartiteMessagePassing):

    def __init__(self, aggr_fn: str = "sum"):
        super(NodeAggregation, self).__init__(aggr_fn)

    def forward(
            self,
            x: torch.Tensor,
            index: torch.LongTensor
    ):
        assert x.size(0) == index.size(0)
        return self.aggregate(x, index)

    def message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
    ) -> torch.Tensor:
        pass  # Not applicable here

    def update(self, x_dst: torch.Tensor, aggr_messages: torch.Tensor) -> torch.Tensor:
        pass  # Not applicable here


class AttentionDirectedBipartiteMessagePassing(DirectedBipartiteMessagePassing):

    def __init__(
            self,
            input_src_dim: int,
            input_dst_dim: int,
            input_edge_dim: int,
            output_dim: int,
            heads: int,
            n_layer_message: int = 2,
            n_layer_update: int = 2,
            positional_encoding_method: str = None
    ):
        super(AttentionDirectedBipartiteMessagePassing, self).__init__(aggr_fn="sum")
        assert n_layer_message >= 1
        assert n_layer_update >= 1
        assert output_dim % heads == 0.0
        self.heads = heads
        self.output_dim = output_dim
        self.positional_encoding_method = positional_encoding_method
        self.d = output_dim / heads
        if positional_encoding_method in ["alibi", "alibi_trainable"]:
            input_edge_dim -= 1
            trainable_slope = positional_encoding_method == "alibi_trainable"
            slope_first_term = 1 / (2 ** (8/heads))
            slope_common_ratio = slope_first_term
            slope = slope_first_term * (slope_common_ratio ** torch.arange(heads))
            slope = slope.unsqueeze(0)
            self.slope = nn.Parameter(data=slope, requires_grad=trainable_slope)
        input_dim = input_src_dim + input_dst_dim + input_edge_dim
        self.q = nn.Parameter(data=torch.rand(1, heads, output_dim // heads) * 0.1, requires_grad=True)
        self.k_linear = ResidualStack(input_dim, output_dim, n_layer_message, last_activation=False)
        self.v_linear = ResidualStack(input_dim, output_dim, n_layer_message, last_activation=False)
        self.out_linear = ResidualStack(output_dim, output_dim, n_layer_update, last_activation=True)

    def forward(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            edge_index: Adj
    ):
        return self.propagate(x_src, x_dst, edge_attr, edge_index)

    def message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
    ) -> torch.Tensor:
        if self.positional_encoding_method in ["alibi", "alibi_trainable"]:
            return self._alibi_attention_message(x_src, x_dst, edge_attr, index)
        else:
            return self._standard_attention_message(x_src, x_dst, edge_attr, index)

    def _standard_attention_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
    ) -> torch.Tensor:
        concat_attr = []
        if x_src is not None:
            concat_attr.append(x_src)
        if x_dst is not None:
            concat_attr.append(x_dst)
        if edge_attr is not None:
            concat_attr.append(edge_attr)
        concat_attr = torch.concat(concat_attr, dim=1)

        k = self.k_linear(concat_attr).view(-1, self.heads, self.output_dim // self.heads)
        v = self.v_linear(concat_attr).view(-1, self.heads, self.output_dim // self.heads)

        attention_coef = (self.d ** (1/2)) * torch.sum(self.q * k, dim=-1)
        attention_weights = softmax(attention_coef, index).view(-1, self.heads, 1)

        out = attention_weights * v
        return out.view(-1, self.output_dim).contiguous()

    def _alibi_attention_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
    ) -> torch.Tensor:
        pos = edge_attr[:, :1]
        edge_attr = edge_attr[:, 1:] if edge_attr.size(1) > 1 else None
        concat_attr = []
        if x_src is not None:
            concat_attr.append(x_src)
        if x_dst is not None:
            concat_attr.append(x_dst)
        if edge_attr is not None:
            concat_attr.append(edge_attr)
        concat_attr = torch.concat(concat_attr, dim=1)

        k = self.k_linear(concat_attr).view(-1, self.heads, self.output_dim // self.heads)
        v = self.v_linear(concat_attr).view(-1, self.heads, self.output_dim // self.heads)

        bias = - self.slope * pos
        attention_coef = (self.d ** (1/2)) * torch.sum(self.q * k, dim=-1) + bias
        attention_weights = softmax(attention_coef, index).view(-1, self.heads, 1)

        out = attention_weights * v
        return out.view(-1, self.output_dim).contiguous()

    def update(self, x_dst: torch.Tensor, aggr_messages: torch.Tensor) -> torch.Tensor:
        return self.out_linear(torch.relu(aggr_messages))


class GeneraLightLaneDemandEmbedding(nn.Module):

    def __init__(
            self,
            lane_segment_dim: int,
            lane_dim: int,
            lane_segment_to_lane_edge_dim: int,
            output_dim: int,
            heads: int,
            n_layer_message: int,
            n_layer_update: int
    ):
        super(GeneraLightLaneDemandEmbedding, self).__init__()
        self.lane_demand_embedding = AttentionDirectedBipartiteMessagePassing(
            lane_segment_dim, lane_dim, lane_segment_to_lane_edge_dim, output_dim, heads,
            n_layer_message=n_layer_message, n_layer_update=n_layer_update,
            positional_encoding_method="alibi_trainable")

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
            output_dim: int,
            heads: int,
            n_layer_message: int,
            n_layer_update: int,
            n_layer_output: int
    ):
        super(GeneraLightMovementDemandEmbedding, self).__init__()
        self.downstream_movement_demand_embedding = AttentionDirectedBipartiteMessagePassing(
            lane_dim, movement_dim, lane_to_downstream_movement_edge_dim, output_dim, heads,
            n_layer_message=n_layer_message, n_layer_update=n_layer_update, positional_encoding_method="alibi_trainable")
        self.upstream_movement_demand_embedding = AttentionDirectedBipartiteMessagePassing(
            lane_dim, movement_dim, lane_to_upstream_movement_edge_dim, output_dim, heads,
            n_layer_message=n_layer_message, n_layer_update=n_layer_update, positional_encoding_method="alibi_trainable")
        self.combined_movement_demand_embedding = ResidualStack(2*output_dim, output_dim, n_layer_output,
                                                                last_activation=True)

    def forward(
            self,
            lane_x: torch.Tensor,
            movement_x: torch.Tensor,
            lane_to_downstream_movement_edge_attr: torch.Tensor,
            lane_to_upstream_movement_edge_attr: torch.Tensor,
            lane_to_downstream_movement_edge_index: Adj,
            lane_to_upstream_movement_edge_index: Adj
    ):
        downstream_embedding = self.downstream_movement_demand_embedding(
            lane_x, movement_x, lane_to_downstream_movement_edge_attr, lane_to_downstream_movement_edge_index)
        upstream_embedding = self.upstream_movement_demand_embedding(
            lane_x, movement_x, lane_to_upstream_movement_edge_attr, lane_to_upstream_movement_edge_index)
        combined_embedding = torch.cat([downstream_embedding, upstream_embedding], dim=-1)
        return self.combined_movement_demand_embedding(combined_embedding)


class GeneraLightPhaseDemandEmbedding(nn.Module):

    def __init__(
            self,
            movement_dim: int,
            phase_dim: int,
            movement_to_phase_edge_dim: int,
            phase_to_phase_edge_dim: int,
            output_dim: int,
            heads: int,
            n_layer_message: int,
            n_layer_update: int
    ):
        super(GeneraLightPhaseDemandEmbedding, self).__init__()
        self.phase_demand_embedding = AttentionDirectedBipartiteMessagePassing(
            movement_dim, phase_dim, movement_to_phase_edge_dim, output_dim, heads,
            n_layer_message=n_layer_message, n_layer_update=n_layer_update, positional_encoding_method=None)
        self.phase_competition = AttentionDirectedBipartiteMessagePassing(
            output_dim, output_dim, phase_to_phase_edge_dim, output_dim, heads,
            n_layer_message=n_layer_message, n_layer_update=n_layer_update, positional_encoding_method=None
        )

    def forward(self, movement_x: torch.Tensor, phase_x: torch.Tensor, movement_to_phase_edge_attr: torch.Tensor,
                phase_to_phase_edge_attr: torch.Tensor, movement_to_phase_edge_index: Adj,
                phase_to_phase_edge_index: Adj):
        phase_demand = self.phase_demand_embedding(movement_x, phase_x, movement_to_phase_edge_attr,
                                                   movement_to_phase_edge_index)
        return self.phase_competition(phase_demand, phase_demand, phase_to_phase_edge_attr, phase_to_phase_edge_index)


class FRAPPhaseDemandEmbedding(nn.Module):

    def __init__(self):
        super(FRAPPhaseDemandEmbedding, self).__init__()
        self.sum_aggregation = NodeAggregation(aggr_fn="sum")

    def forward(self, movement_x: torch.Tensor, movement_to_phase_edge_index: Adj) -> torch.Tensor:
        return self.sum_aggregation(movement_x[movement_to_phase_edge_index[0]],
                                    movement_to_phase_edge_index[1])


class FRAPPhasePairEmbedding(nn.Module):

    def __init__(self, phase_dim: int):
        super(FRAPPhasePairEmbedding, self).__init__()
        self.pair_relation_embedding = nn.Embedding(num_embeddings=2, embedding_dim=phase_dim)

    def forward(self, phase_demand_embedding: torch.Tensor, pair_partial_competing: torch.LongTensor,
                pair_edge_index: Adj) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_relation_embedding = self.pair_relation_embedding(pair_partial_competing.squeeze())
        pair_demand_embedding = torch.cat([phase_demand_embedding[pair_edge_index[1]],
                                           phase_demand_embedding[pair_edge_index[0]]], dim=1)
        return pair_relation_embedding, pair_demand_embedding


class FRAPPhaseCompetition(nn.Module):

    def __init__(self, phase_dim: int, n_layer: int):
        super(FRAPPhaseCompetition, self).__init__()
        self.pair_demand_linear_stack = LinearStack(2*phase_dim, phase_dim, phase_dim, n_layer, last_activation=True)
        self.pairwise_competition_linear = nn.Linear(phase_dim, 1)
        self.sum_aggregation = NodeAggregation(aggr_fn="sum")

    def forward(self, pair_relation_embedding: torch.Tensor, pair_demand_embedding: torch.Tensor, pair_edge_index: Adj):
        pair_demand_representation = self.pair_demand_linear_stack(pair_demand_embedding)
        pair_competition_representation = pair_demand_representation * pair_relation_embedding
        return self.sum_aggregation(pair_competition_representation, pair_edge_index[1])
        #pairwise_competition_result = self.pairwise_competition_linear(pair_competition_representation)
        #return phase_score

from abc import abstractmethod
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch_geometric.nn import aggr
from torch_geometric.typing import Adj
from torch_geometric.data import HeteroData

from src.modules.utils import concat_features, neighborhood_attention
from src.modules.base_modules import ResidualBlock, ResidualStack


class MessagePassing(nn.Module):

    def __init__(self, agg_fct: str = "sum"):
        super(MessagePassing, self).__init__()
        self.agg_fct = None
        if agg_fct == "sum":
            self.agg_fct = aggr.SumAggregation()
        elif agg_fct == "mean":
            self.agg_fct = aggr.MeanAggregation()
        elif agg_fct == "max":
            self.agg_fct = aggr.MaxAggregation()
        elif agg_fct == "min":
            self.agg_fct = aggr.MinAggregation()

    @abstractmethod
    def message(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor,
                pos: torch.LongTensor = None) -> torch.Tensor:
        pass

    def aggregate(self, x: torch.Tensor, index: torch.LongTensor, dim_size: int = None) -> torch.Tensor:
        return self.agg_fct(x, index, dim=0, dim_size=dim_size)

    @abstractmethod
    def update(self, x_dst: torch.Tensor, aggr_message: torch.Tensor) -> torch.Tensor:
        pass


class HeteroNeighborhoodAttention(MessagePassing):

    def __init__(
            self,
            edge_types: List[Tuple[str, str, str]],
            node_dims: Dict,
            edge_dims: Dict,
            output_dim: int,
            heads: int,
            skip_connection: bool = False,
            dropout_prob: float = 0.0
    ):
        super(HeteroNeighborhoodAttention, self).__init__(agg_fct="sum")
        dst_node = list(set([edge_type[2] for edge_type in edge_types]))
        assert len(dst_node) == 1
        self.dst_node = dst_node[0]
        dst_dim = node_dims[self.dst_node]
        self.node_dims, self.edge_dims = node_dims, edge_dims
        self.msg_fct = None
        self.msg_fcts = nn.ModuleDict()
        message_dim = 0
        for edge in edge_types:
            src_node = edge[0]
            src_dim = node_dims[src_node]
            edge_dim = edge_dims[edge]
            self.msg_fcts["|".join(edge)] = AttentionMessage(src_dim, dst_dim, edge_dim, output_dim, heads,
                                                             dropout_prob)
            message_dim += output_dim
        self.upd_fct = ResidualBlock(message_dim, output_dim, last_activation=not skip_connection,
                                     dropout_prob=dropout_prob)
        self.skip_connection = skip_connection
        if skip_connection:
            self.identity = nn.Identity() if dst_dim == output_dim else nn.Linear(dst_dim, output_dim, bias=False)

    def forward(self, graph: HeteroData):
        x_dst = graph[self.dst_node].x
        all_agg_messages = []
        for edge, msg_fct in self.msg_fcts.items():
            self.msg_fct = edge
            edge = tuple(edge.split("|"))
            edge_index = graph[edge[0], edge[1], edge[2]].edge_index
            index = edge_index[1]
            src_node, dst_node = edge[0], edge[2]
            x_src = graph[src_node].x
            x_src_expanded = None
            if self.node_dims[src_node] > 0:
                x_src_expanded = x_src[edge_index[0]]
            x_dst_expanded = None
            if self.node_dims[dst_node] > 0:
                x_dst_expanded = x_dst[edge_index[1]]
            edge_attr = None
            if self.edge_dims[edge] > 0:
                edge_attr = graph[edge[0], edge[1], edge[2]].edge_attr
            message = self.message(x_src_expanded, x_dst_expanded, edge_attr, index)
            agg_messages = self.aggregate(message, index, dim_size=x_dst.size(0))
            all_agg_messages.append(agg_messages)
        agg_messages = torch.cat(all_agg_messages, dim=-1)
        return self.update(x_dst, agg_messages)

    def message(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor,
                pos: torch.LongTensor = None) -> torch.Tensor:
        return self.msg_fcts[self.msg_fct](x_src, x_dst, edge_attr, index)

    def update(self, x_dst: torch.Tensor, aggr_message: torch.Tensor) -> torch.Tensor:
        x_dst_upd = self.upd_fct(aggr_message)
        return torch.relu(self.identity(x_dst) + x_dst_upd) if self.skip_connection else x_dst_upd


class AttentionMessage(nn.Module):

    def __init__(
            self,
            src_dim: int,
            dst_dim: int,
            edge_dim: int,
            output_dim: int,
            heads: int,
            dropout_prob: float = 0.0
    ):
        super(AttentionMessage, self).__init__()
        input_dim = src_dim + dst_dim + edge_dim
        self.heads = heads
        self.q = nn.Parameter(data=torch.rand(1, output_dim) * 0.1, requires_grad=True)
        self.k_fct = ResidualBlock(input_dim, output_dim, last_activation=False, dropout_prob=dropout_prob)
        self.v_fct = ResidualBlock(input_dim, output_dim, last_activation=False, dropout_prob=dropout_prob)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor):
        x = concat_features([x_src, x_dst, edge_attr])
        q, k, v = self.q, self.k_fct(x), self.v_fct(x)
        return neighborhood_attention(q, k, v, self.heads, index)


class NeighborhoodAttention(MessagePassing):

    def __init__(
            self,
            src_dim: int,
            dst_dim: int,
            edge_dim: int,
            output_dim: int,
            heads: int,
            n_residuals: int = 2,
            positional_encoding_method: str = None,
            skip_connection: bool = False,
            dropout_prob: float = 0.0
    ):
        super(NeighborhoodAttention, self).__init__(agg_fct="sum")
        assert n_residuals >= 1
        assert output_dim % heads == 0.0
        self.heads = heads
        self.output_dim = output_dim
        self.positional_encoding_method = positional_encoding_method
        self.d = output_dim / heads
        input_dim = src_dim + dst_dim + edge_dim
        if positional_encoding_method in ["alibi", "alibi_learnable"]:
            learnable_slope = positional_encoding_method == "alibi_learnable"
            slope_first_term = 1 / (2 ** (8/heads))
            slope_common_ratio = slope_first_term
            slope = slope_first_term * (slope_common_ratio ** torch.arange(heads))
            slope = slope.unsqueeze(0)
            self.slope = nn.Parameter(data=slope, requires_grad=learnable_slope)
        elif positional_encoding_method == "learnable":
            edge_dim = edge_dim - 1
            input_dim = src_dim + dst_dim + edge_dim
            max_len = 10
            self.input_layer = ResidualStack(input_dim, output_dim, 1, dropout_prob=dropout_prob)
            self.pos_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=output_dim)
            self.pos_embedding.weight.data.uniform_(-0.1, 0.1)
            input_dim = output_dim

        self.q = nn.Parameter(data=torch.rand(1, output_dim) * 0.1, requires_grad=True)
        self.k_layers = ResidualStack(input_dim, output_dim, n_residuals, last_activation=False,
                                      dropout_prob=dropout_prob)
        self.v_layers = ResidualStack(input_dim, output_dim, n_residuals, last_activation=False,
                                      dropout_prob=dropout_prob)
        self.out_layers = ResidualStack(output_dim, output_dim, n_residuals, last_activation=not skip_connection,
                                        dropout_prob=dropout_prob)
        if skip_connection:
            self.identity = nn.Identity() if dst_dim == output_dim else nn.Linear(dst_dim, output_dim, bias=False)
        self.skip_connection = skip_connection

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, edge_index: Adj,
                pos: torch.LongTensor = None) -> torch.Tensor:
        dim_size = x_dst.size(0) if x_dst is not None else None
        x_src_expanded, x_dst_expanded = None, None
        if x_src is not None:
            x_src_expanded = x_src[edge_index[0]]
        if x_dst is not None:
            x_dst_expanded = x_dst[edge_index[1]]
        index = edge_index[1]
        messages = self.message(x_src_expanded, x_dst_expanded, edge_attr, index, pos)
        aggr_messages = self.aggregate(messages, index, dim_size=dim_size)
        return self.update(x_dst, aggr_messages)

    def message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor,
            pos: torch.LongTensor = None
    ) -> torch.Tensor:
        if self.positional_encoding_method in ["alibi", "alibi_learnable"]:
            return self._alibi_message(x_src, x_dst, edge_attr, index, pos)
        if self.positional_encoding_method in ["sinusoidal"]:
            return self._sinusoidal_positional_encoding_message(x_src, x_dst, edge_attr, index)
        if self.positional_encoding_method == "learnable":
            return self._learnable_positional_encoding_message(x_src, x_dst, edge_attr, index)
        else:
            return self._standard_attention_message(x_src, x_dst, edge_attr, index)

    def _standard_attention_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor,
            pos: torch.LongTensor = None
    ) -> torch.Tensor:
        x = concat_features([x_src, x_dst, edge_attr])
        q, k, v = self.q, self.k_layers(x), self.v_layers(x)
        return neighborhood_attention(q, k, v, self.heads, index)

    def _alibi_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor,
            pos: torch.LongTensor = None
    ) -> torch.Tensor:
        pos = pos.unsqueeze(-1)
        x = concat_features([x_src, x_dst, edge_attr])
        q, k, v = self.q, self.k_layers(x), self.v_layers(x)
        bias = - self.slope * pos
        return neighborhood_attention(q, k, v, self.heads, index, bias=bias)

    def _sinusoidal_positional_encoding_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor,
            pos: torch.LongTensor = None
    ) -> torch.Tensor:
        pass

    def _learnable_positional_encoding_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor,
            pos: torch.LongTensor = None
    ) -> torch.Tensor:
        pos = edge_attr[:, :1].squeeze().to(dtype=torch.long)
        max_len = self.pos_embedding.num_embeddings
        pos[pos > max_len-1] = max_len-1
        edge_attr = edge_attr[:, 1:]
        x = concat_features([x_src, x_dst, edge_attr])
        x = self.input_layer(x)
        pos_embedding = self.pos_embedding(pos)
        x = x + pos_embedding
        q, k, v = self.q, self.k_layers(x), self.v_layers(x)
        return neighborhood_attention(q, k, v, self.heads, index)

    def update(self, x_dst: torch.Tensor, aggr_messages: torch.Tensor) -> torch.Tensor:
        out = self.out_layers(torch.relu(aggr_messages))
        out = torch.relu(self.identity(x_dst) + out) if self.skip_connection else out
        return out


class NeighborhoodAggregation(MessagePassing):

    def __init__(
            self,
            src_dim: int,
            dst_dim: int,
            edge_dim: int,
            output_dim: int,
            n_residuals: int,
            skip_connection: bool = False,
            dropout_prob: float = 0.0
    ):
        super(NeighborhoodAggregation, self).__init__(agg_fct="sum")
        input_dim = src_dim + dst_dim + edge_dim
        self.msg_fct = ResidualStack(input_dim, output_dim, n_residuals, dropout_prob=dropout_prob)
        self.upd_fct = ResidualStack(output_dim, output_dim, n_residuals, dropout_prob=dropout_prob)
        if skip_connection:
            self.identity = nn.Identity() if dst_dim == output_dim else nn.Linear(dst_dim, output_dim)
        self.skip_connection = skip_connection

    def message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor,
            pos: torch.LongTensor = None
    ) -> torch.Tensor:
        x = concat_features([x_src, x_dst, edge_attr])
        msg = self.msg_fct(x)
        return msg

    def update(self, x_dst: torch.Tensor, agg_msg: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x_dst) if self.skip_connection else torch.zeros_like(x_dst)
        return identity + self.upd_fct(agg_msg)

from abc import abstractmethod
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import aggr
from torch_geometric.typing import Adj

from src.modules.utils import concat_features, neighborhood_attention
from src.modules.base_modules import ResidualStack


class MessagePassing(nn.Module):

    def __init__(self, aggr_fn: str = "sum"):
        super(MessagePassing, self).__init__()
        self.aggr_fn = None
        if aggr_fn == "sum":
            self.aggr_fn = aggr.SumAggregation()
        elif aggr_fn == "mean":
            self.aggr_fn = aggr.MeanAggregation()
        elif aggr_fn == "max":
            self.aggr_fn = aggr.MaxAggregation()
        elif aggr_fn == "min":
            self.aggr_fn = aggr.MinAggregation()

    @abstractmethod
    def forward(self, *args: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def propagate(self, *args: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def message(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor) \
            -> torch.Tensor:
        pass

    def aggregate(self, x: torch.Tensor, index: torch.LongTensor, dim_size: int = None) -> torch.Tensor:
        return self.aggr_fn(x, index, dim=0, dim_size=dim_size)

    @abstractmethod
    def update(self, x_dst: torch.Tensor, aggr_message: torch.Tensor) -> torch.Tensor:
        pass


class HeteroMessagePassing(MessagePassing):

    def __init__(self, aggr_fn: str = "sum"):
        super(HeteroMessagePassing, self).__init__(aggr_fn)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, edge_index: Adj)\
            -> torch.Tensor:
        return self.propagate(x_src, x_dst, edge_attr, edge_index)

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
    def message(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor) \
            -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, x_dst: torch.Tensor, aggr_message: torch.Tensor) -> torch.Tensor:
        pass


class HomoMessagePassing(MessagePassing):

    def __init__(self, aggr_fn: str = "sum"):
        super(HomoMessagePassing, self).__init__(aggr_fn=aggr_fn)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        return self.propagate(x, edge_attr, edge_index)

    def propagate(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        dim_size = x.size(0)
        x_src, x_dst = x, x
        x_src_expanded, x_dst_expanded = x[edge_index[0]], x[edge_index[1]]
        index = edge_index[1]
        messages = self.message(x_src_expanded, x_dst_expanded, edge_attr, index)
        aggr_messages = self.aggregate(messages, index, dim_size=dim_size)
        return self.update(x_dst, aggr_messages)

    @abstractmethod
    def message(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor) \
            -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, x_dst: torch.Tensor, aggr_message: torch.Tensor) -> torch.Tensor:
        pass


class HeteroNeighborhoodAttention(HeteroMessagePassing):

    def __init__(
            self,
            src_dim: int,
            dst_dim: int,
            edge_dim: int,
            output_dim: int,
            heads: int,
            n_residuals: int = 2,
            positional_encoding_method: str = None,
            skip_connection: bool = False
    ):
        super(HeteroNeighborhoodAttention, self).__init__(aggr_fn="sum")
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
            input_dim -= 1
        elif positional_encoding_method == "learnable":
            edge_dim = edge_dim - 1
            input_dim = src_dim + dst_dim + edge_dim
            max_len = 10
            self.input_layer = ResidualStack(input_dim, output_dim, 1)
            self.pos_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=output_dim)
            self.pos_embedding.weight.data.uniform_(-0.1, 0.1)
            input_dim = output_dim

        self.q = nn.Parameter(data=torch.rand(1, output_dim) * 0.1, requires_grad=True)
        self.k_layers = ResidualStack(input_dim, output_dim, n_residuals, last_activation=False)
        self.v_layers = ResidualStack(input_dim, output_dim, n_residuals, last_activation=False)
        self.out_layers = ResidualStack(output_dim, output_dim, n_residuals, last_activation=not skip_connection)
        if skip_connection:
            self.identity = nn.Identity() if dst_dim == output_dim else nn.Linear(dst_dim, output_dim, bias=False)
        self.skip_connection = skip_connection

    def message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
    ) -> torch.Tensor:
        if self.positional_encoding_method in ["alibi", "alibi_learnable"]:
            return self._alibi_message(x_src, x_dst, edge_attr, index)
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
            index: torch.LongTensor
    ) -> torch.Tensor:
        x = concat_features([x_src, x_dst, edge_attr])
        q, k, v = self.q, self.k_layers(x), self.v_layers(x)
        return neighborhood_attention(q, k, v, self.heads, index)

    def _alibi_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
    ) -> torch.Tensor:
        pos = edge_attr[:, :1]
        edge_attr = edge_attr[:, 1:] if edge_attr.size(1) > 1 else None
        x = concat_features([x_src, x_dst, edge_attr])
        q, k, v = self.q, self.k_layers(x), self.v_layers(x)
        bias = - self.slope * pos
        return neighborhood_attention(q, k, v, self.heads, index, bias=bias)

    def _sinusoidal_positional_encoding_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
    ) -> torch.Tensor:
        pass

    def _learnable_positional_encoding_message(
            self,
            x_src: torch.Tensor,
            x_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            index: torch.LongTensor
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


class HomoNeighborhoodAttention(HomoMessagePassing):

    def __init__(
            self,
            input_dim: int,
            edge_dim: int,
            output_dim: int,
            heads: int,
            n_residuals: int = 2,
            skip_connection: bool = True
    ):
        super(HomoMessagePassing, self).__init__(aggr_fn="sum")
        combined_input_dim = 2 * input_dim + edge_dim
        self.heads = heads
        self.q = nn.Parameter(data=torch.rand(1, output_dim) * 0.1, requires_grad=True)
        self.k_layers = ResidualStack(combined_input_dim, output_dim, n_residuals, last_activation=False)
        self.v_layers = ResidualStack(combined_input_dim, output_dim, n_residuals, last_activation=False)
        self.out_layers = ResidualStack(output_dim, output_dim, n_residuals, last_activation=not skip_connection)
        if skip_connection:
            self.identity = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim, bias=False)
        self.skip_connection = skip_connection

    def message(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor, index: torch.LongTensor) \
            -> torch.Tensor:
        x = concat_features([x_src, x_dst, edge_attr])
        q, k, v = self.q, self.k_layers(x), self.v_layers(x)
        return neighborhood_attention(q, k, v, self.heads, index)

    def update(self, x_dst: torch.Tensor, aggr_message: torch.Tensor) -> torch.Tensor:
        out = self.out_layers(torch.relu(aggr_message))
        out = torch.relu(self.identity(x_dst) + out) if self.skip_connection else out
        return out

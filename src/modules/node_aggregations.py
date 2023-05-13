from abc import abstractmethod
import sys
from typing import Tuple, Dict

import torch
from torch_geometric import nn as pyg_nn
from torch_geometric.typing import Adj


class NodeAggregation(pyg_nn.MessagePassing):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, NodeAggregation)
        return obj

    def __init__(self, aggr: str):
        super(NodeAggregation, self).__init__(aggr=aggr)

    @abstractmethod
    def forward(self, x: torch.Tensor, index: torch.LongTensor = None, edge_index: Adj = None) -> torch.Tensor:
        pass

    def _update_x_and_edge_index(self,
                                 x_src: torch.Tensor,
                                 x_dest: torch.Tensor = None,
                                 index: torch.LongTensor = None,
                                 edge_index: Adj = None) -> Tuple[torch.Tensor, Adj, int]:
        assert index is not None or edge_index is not None
        device = x_src.get_device()
        if index is not None:
            src_node_ids = torch.arange(0, index.size(0), device=device).unsqueeze(0)
            dest_node_ids = index.unsqueeze(0)
            edge_index = torch.cat([src_node_ids, dest_node_ids], dim=0)
        offset = x_src.size(0)
        n_src_nodes = x_src.size(0)
        n_dest_nodes = torch.max(edge_index[1]) + 1
        dim_src_nodes = x_src.size(1)
        if x_dest is not None and x_dest.dim() == 1:
            x_dest = x_dest.repeat(n_dest_nodes, 1)
        else:
            x_dest = torch.zeros(n_dest_nodes, dim_src_nodes, device=device)
        x = torch.cat([x_src, x_dest], dim=0)
        edge_index[1] += n_src_nodes
        return x, edge_index, offset


class SimpleNodeAggregation(NodeAggregation):

    def __init__(self, aggr: str):
        super(SimpleNodeAggregation, self).__init__(aggr)

    def forward(self, x: torch.Tensor, index: torch.LongTensor = None, edge_index: Adj = None) -> torch.Tensor:
        x, edge_index, offset = self._update_x_and_edge_index(x, index=index, edge_index=edge_index)
        x = self.propagate(edge_index, x=x)
        return x[offset:]

    @staticmethod
    def message(x_j: torch.Tensor) -> torch.Tensor:
        return x_j

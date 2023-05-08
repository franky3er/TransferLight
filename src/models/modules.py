from typing import Dict, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch import Tensor
from torch_geometric import nn as pyg_nn
from torch_geometric.typing import Adj, NodeType

from src.params import DEVICE


class HeteroModule(nn.Module):

    def __init__(self, modules: nn.ModuleDict):
        super(HeteroModule, self).__init__()
        self.modules = modules

    def forward(self, x_dict: Dict[NodeType, torch.Tensor]):
        for node_type, module in self.modules.items():
            x_dict[node_type] = module(x_dict[node_type])
        return x_dict


class NodeAggregation(pyg_nn.MessagePassing):

    def __init__(self, aggr_fn: str = "mean"):
        super().__init__(aggr=aggr_fn)

    def forward(self, x: torch.Tensor, edge_index: Adj):
        x, edge_index = self._update_x_and_edge_index(x, edge_index)
        print(x)
        print(edge_index)

    @staticmethod
    def _update_x_and_edge_index(x_src: torch.Tensor, edge_index: Adj) -> Tuple[torch.Tensor, Adj]:
        n_src_nodes = torch.max(edge_index[0]) + 1
        n_dest_nodes = torch.max(edge_index[1]) + 1
        dim_src_nodes = x_src.size(1)
        x_dest = torch.zeros(n_dest_nodes, dim_src_nodes, device=DEVICE)
        x = torch.cat([x_src, x_dest], dim=0)
        edge_index[1] += n_src_nodes
        return x, edge_index


class ArgmaxAggregation:

    def __call__(self, x: Tensor, index: Tensor, return_indices: bool = False):
        group_elements = defaultdict(lambda: [])
        group_indices = defaultdict(lambda: [])
        for idx, (group, element) in enumerate(zip(index, x)):
            group_elements[group.item()].append(element)
            group_indices[group.item()].append(idx)
        group_elements = {group: torch.cat(elements) for group, elements in group_elements.items()}
        group_indices = {group: torch.tensor(indices, device=x.device) for group, indices in group_indices.items()}
        group_elements = dict(sorted(group_elements.items()))
        group_indices = dict(sorted(group_indices.items()))
        group_argmax = {group: torch.argmax(elements) for group, elements in group_elements.items()}
        group_argmax_indices = {group: group_indices[group][argmax] for group, argmax in group_argmax.items()}
        if return_indices:
            return torch.cat([idx.unsqueeze(dim=0) for idx in group_argmax_indices.values()])
        return torch.cat([argmax.unsqueeze(dim=0) for argmax in group_argmax.values()])


class NumElementsAggregation:

    def __call__(self, index: Tensor):
        groups = defaultdict(lambda: 0)
        for idx in index:
            groups[idx.item()] += 1
        groups = dict(sorted(groups.items()))
        return torch.cat([torch.tensor(n, device=index.device).unsqueeze(0) for n in groups.values()])


class SimpleConv(pyg_nn.MessagePassing):

    def __init__(self, aggr: str = "mean"):
        super().__init__(aggr=aggr)

    def forward(self, x: Tensor, edge_index: Adj):
        return self.propagate(edge_index, x=x)


class PhaseDemandLayer(pyg_nn.MessagePassing):

    def __init__(self, movement_dim: int, phase_dim: int):
        super(PhaseDemandLayer, self).__init__(aggr="mean")
        self.phase_demand = nn.Sequential(
            nn.Linear(in_features=movement_dim, out_features=phase_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=phase_dim, out_features=phase_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, edge_index: Tensor):
        aggregated_movements = self.propagate(edge_index, x=x)
        return self.phase_demand(aggregated_movements)


class PhaseCompetitionLayer(pyg_nn.MessagePassing):

    def __init__(self, phase_dim: int):
        super(PhaseCompetitionLayer, self).__init__(aggr="mean")
        self.pair_demand_embedding = nn.Sequential(
            nn.Linear(in_features=2*phase_dim, out_features=phase_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=phase_dim, out_features=phase_dim),
            nn.ReLU(inplace=True)
        )
        self.pair_relation_embedding = nn.Linear(in_features=2, out_features=phase_dim, bias=False)
        self.phase_competition_layer = nn.Sequential(
            nn.Linear(in_features=phase_dim, out_features=phase_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=phase_dim, out_features=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor = None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor):
        pair_demand_embedding = self.pair_demand_embedding(torch.cat([x_i, x_j], dim=1))
        #pair_relation_embedding = self.pair_relation_embedding(edge_attr) if edge_attr is not None else 1
        #return self.phase_competition_layer(pair_demand_embedding * pair_relation_embedding)
        return self.phase_competition_layer(pair_demand_embedding)


class FlexibleDuelingHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        self.state_value_stream = nn.Sequential

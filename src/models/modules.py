from typing import Dict, Union

import torch
from torch import nn
from torch import Tensor
from torch_geometric import nn as pyg_nn
from torch_geometric.typing import Adj, NodeType, OptPairTensor


class HeteroModule(nn.Module):

    def __init__(self, modules: Dict[NodeType, nn.Module]):
        super(HeteroModule, self).__init__()
        self.modules = modules

    def forward(self, x_dict: Dict[NodeType, torch.Tensor]):
        for node_type, module in self.modules.items():
            x_dict[node_type] = module(x_dict[node_type])
        return x_dict


class PhaseDemandLayer(pyg_nn.MessagePassing):

    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, x: Tensor, edge_index: Adj):
        return self.propagate(edge_index, x=x)


class PhaseCompetitionLayer(pyg_nn.MessagePassing):

    def __init__(self, phase_dim: int):
        super(PhaseCompetitionLayer, self).__init__(aggr="sum")
        self.pair_demand_embedding = nn.Sequential(
            nn.Linear(in_features=2*phase_dim, out_features=phase_dim),
            nn.ReLU(inplace=True)
        )
        self.pair_relation_embedding = nn.Linear(in_features=2, out_features=phase_dim, bias=False)
        self.phase_competition_layer = nn.Sequential(
            nn.Linear(in_features=phase_dim, out_features=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor = None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor):
        if edge_attr is None:
            edge_attr = torch.ones(x_i.size(0), 2)
        pair_demand_embedding = self.pair_demand_embedding(torch.cat([x_i, x_j], dim=1))
        pair_relation_embedding = self.pair_relation_embedding(edge_attr)
        return self.phase_competition_layer(pair_demand_embedding * pair_relation_embedding)

from typing import Dict, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch import Tensor
from torch_geometric import nn as pyg_nn
from torch_geometric.typing import Adj, NodeType


class HeteroModule(nn.Module):

    def __init__(self, modules: nn.ModuleDict):
        super(HeteroModule, self).__init__()
        self.modules = modules

    def forward(self, x_dict: Dict[NodeType, torch.Tensor]):
        for node_type, module in self.modules.items():
            x_dict[node_type] = module(x_dict[node_type])
        return x_dict


class MultiInputSequential(nn.Sequential):

    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class FlexibleArgmax(nn.Module):

    @staticmethod
    def forward(x: torch.Tensor, group_index: torch.Tensor, return_argmax_indices: bool = False,
                keepdim: bool = True):
        x = x.squeeze()
        assert x.size(0) == group_index.size(0)
        device = x.get_device()
        dtype = x.dtype
        n_items = x.size(0)
        n_groups = torch.max(group_index) + 1
        x_repeated = x.unsqueeze(1).repeat(1, n_groups)
        item_index = torch.arange(0, n_items, device=device)
        group_index = torch.nn.functional.one_hot(group_index)
        cumsum_group_index = torch.cumsum(group_index, dim=0) - 1
        dummy_tensor = torch.ones(n_items, n_groups, dtype=dtype, device=device) * torch.finfo(dtype).min
        dummy_tensor = (1 - group_index) * dummy_tensor + group_index * x_repeated
        argmax_index = torch.argmax(dummy_tensor, dim=0, keepdim=True)
        argmax_values = torch.gather(cumsum_group_index, 0, argmax_index).squeeze()
        argmax_indices = item_index[argmax_index.squeeze()]
        if argmax_values.dim() == 0 and keepdim:
            argmax_values = argmax_values.unsqueeze(0)
            argmax_indices = argmax_indices.unsqueeze(0)
        if return_argmax_indices:
            return argmax_values, argmax_indices
        return argmax_values.squeeze()


class ResidualStack(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, stack_size: int, last_activation: bool = True):
        super(ResidualStack, self).__init__()
        residual_blocks = []
        for i in range(stack_size - 1):
            residual_blocks.append(ResidualBlock(input_dim if i == 0 else output_dim, output_dim))
        residual_blocks.append(ResidualBlock(input_dim if stack_size == 1 else output_dim, output_dim, last_activation))
        self.residual_blocks = nn.Sequential(*tuple(residual_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_blocks(x)


class ResidualBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, last_activation: bool = True):
        super(ResidualBlock, self).__init__()
        self.identity = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim, bias=False)
        self.residual_function = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.last_activation = last_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual_function(x) + self.identity(x)
        return torch.relu(out) if self.last_activation else out


class LinearStack(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layer: int,
                 last_activation: bool = True,
                 residual_connections: bool = False):
        super(LinearStack, self).__init__()
        self.layers = (self._get_linear_stack(input_dim, hidden_dim, output_dim, n_layer, last_activation)
                       if not residual_connections
                       else self._get_res_linear_stack(input_dim, hidden_dim, output_dim, n_layer, last_activation))

    @staticmethod
    def _get_linear_stack(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, last_activation: bool):
        layers = []
        for layer in range(n_layers):
            if layer == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                #layers.append(nn.LayerNorm(hidden_dim))
            elif layer == n_layers-1:
                layers.append(nn.Linear(hidden_dim, output_dim))
                #layers.append(nn.LayerNorm(output_dim))
                if not last_activation:
                    break
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                #layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*tuple(layers))

    @staticmethod
    def _get_res_linear_stack(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, last_activation: bool):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        return self.layers(x)


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


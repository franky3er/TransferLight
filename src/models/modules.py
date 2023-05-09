from typing import Dict, Tuple
from collections import defaultdict
import itertools

import torch
from torch import nn
from torch import Tensor
import torch_geometric as pyg
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


class FlexibleCategorical(nn.Module):

    def __init__(self, logits: torch.Tensor, index: torch.Tensor):
        super(FlexibleCategorical, self).__init__()
        self.logits = logits
        self.index = index
        self.log_prob_fn = FlexibleCategoricalLogProb()
        self.log_prob_val = None
        self.sampling_fn = FlexibleCategoricalSampler()
        self.entropy_fn = FlexibleCategoricalEntropy()
        self.entropy_val = None

    def log_prob(self, values: torch.Tensor = None):
        if self.log_prob_val is None:
            self.log_prob_val = self.log_prob_fn(self.logits, self.index)
        return self.log_prob_val if values is None else self.log_prob_val[values]

    def entropy(self):
        if self.entropy_val is None:
            self.entropy_val = self.entropy_fn(self.logits, self.index)
        return self.entropy_val

    def sample(self, return_sample_indices: bool = False):
        return self.sampling_fn(self.logits, self.index, return_sample_indices=return_sample_indices)

    def to(self, device):
        self.log_prob_fn.to(device)
        self.sampling_fn.to(device)
        self.entropy_fn.to(device)
        return super(FlexibleCategorical, self).to(device)


class FlexibleCategoricalLogProb(pyg_nn.MessagePassing):

    def __init__(self):
        super(FlexibleCategoricalLogProb, self).__init__(aggr="sum")
        self.device = "cpu"

    def forward(self, logits: torch.Tensor, index: torch.LongTensor):
        logits = logits.unsqueeze(-1)
        edge_index = self._create_edge_index(index)
        exp_logits = torch.exp(logits)
        return self.propagate(edge_index, logits=logits, exp_logits=exp_logits).squeeze()

    def _create_edge_index(self, index: torch.Tensor):
        dist_logits = [[] for _ in range(torch.max(index) + 1)]
        for logit_idx, dist in enumerate(index):
            dist_logits[dist].append(logit_idx)
        edge_index = None
        for dist_logit_indices in dist_logits:
            src_nodes, dest_nodes = [], []
            for dest_node, src_node in itertools.product(dist_logit_indices, repeat=2):
                src_nodes.append(src_node)
                dest_nodes.append(dest_node)
            edge_index_dist = torch.tensor([src_nodes, dest_nodes], device=self.device)
            edge_index = edge_index_dist if edge_index is None else torch.cat([edge_index, edge_index_dist], dim=1)
        return edge_index

    @staticmethod
    def message(exp_logits_j: torch.Tensor) -> torch.Tensor:
        return exp_logits_j

    @staticmethod
    def update(aggregated_message: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return logits - torch.log(aggregated_message)

    def to(self, device):
        self.device = device
        return super(FlexibleCategoricalLogProb, self).to(device)


class FlexibleCategoricalEntropy(pyg_nn.MessagePassing):

    def __init__(self):
        super(FlexibleCategoricalEntropy, self).__init__(aggr="sum")
        self.device = "cpu"

    def forward(self, logits: torch.Tensor, index: torch.Tensor):
        n_groups = torch.max(index) + 1
        edge_index = self._create_edge_index(index)
        probs = pyg.utils.softmax(logits, index).unsqueeze(1)
        log_probs = torch.log(probs)
        entropy = self.propagate(edge_index, probs=probs, log_probs=log_probs).squeeze()[:n_groups]
        return entropy

    def _create_edge_index(self, index: torch.Tensor):
        src_nodes = torch.arange(index.size(0), device=self.device).unsqueeze(0)
        dest_nodes = index.unsqueeze(0)
        edge_index = torch.cat([src_nodes, dest_nodes], dim=0)
        return edge_index

    def message(self, probs_j: torch.Tensor, log_probs_j: torch.Tensor):
        return - probs_j * log_probs_j

    def to(self, device):
        self.device = device
        return super(FlexibleCategoricalEntropy, self).to(device)


class FlexibleCategoricalSampler(nn.Module):

    def __init__(self):
        super(FlexibleCategoricalSampler, self).__init__()
        self.argmax = FlexibleArgmax()
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(loc=0, scale=1)
        self.device = "cpu"

    def forward(self, logits: torch.Tensor, index: torch.Tensor, return_sample_indices: bool = False):
        # Gumbel-max trick
        z = self.gumbel_dist.sample(logits.shape).to(self.device)
        samples = self.argmax(logits + z, index, return_argmax_indices=return_sample_indices)
        return samples

    def to(self, device):
        self.device = device
        self.argmax.to(device)
        return super(FlexibleCategoricalSampler, self).to(device)


class FlexibleArgmax(nn.Module):

    def __init__(self):
        super(FlexibleArgmax, self).__init__()
        self.device = "cpu"

    def forward(self, x: torch.Tensor, group_index: torch.Tensor, return_argmax_indices: bool = False):
        assert x.size(0) == group_index.size(0)
        dtype = x.dtype
        n_items = x.size(0)
        n_groups = torch.max(group_index) + 1
        x_repeated = x.unsqueeze(1).repeat(1, n_groups)
        item_index = torch.arange(0, n_items, device=self.device)
        group_index = torch.nn.functional.one_hot(group_index)
        cumsum_group_index = torch.cumsum(group_index, dim=0) - 1
        dummy_tensor = torch.ones(n_items, n_groups, dtype=dtype, device=self.device) * torch.finfo(dtype).min
        dummy_tensor = (1 - group_index) * dummy_tensor + group_index * x_repeated
        argmax_index = torch.argmax(dummy_tensor, dim=0, keepdim=True)
        argmax_values = torch.gather(cumsum_group_index, 0, argmax_index).squeeze()
        argmax_indices = item_index[argmax_index.squeeze()]
        if return_argmax_indices:
            return argmax_values, argmax_indices
        return argmax_values.squeeze()

    def to(self, device):
        self.device = device
        return super(FlexibleArgmax, self).to(device)


class NodeAggregation(pyg_nn.MessagePassing):

    def __init__(self, aggr_fn: str = "mean"):
        super(NodeAggregation, self).__init__(aggr=aggr_fn)
        self.device = "cpu"

    def forward(self, x: torch.Tensor, edge_index: Adj):
        n_src_nodes = x.size(0)
        x, edge_index = self._update_x_and_edge_index(x, edge_index)
        x = self.propagate(edge_index, x=x)
        return x[n_src_nodes:]

    def message(self, x_j: torch.Tensor):
        return x_j

    def _update_x_and_edge_index(self, x_src: torch.Tensor, edge_index: Adj) -> Tuple[torch.Tensor, Adj]:
        n_src_nodes = x_src.size(0)
        n_dest_nodes = torch.max(edge_index[1]) + 1
        dim_src_nodes = x_src.size(1)
        x_dest = torch.zeros(n_dest_nodes, dim_src_nodes, device=self.device)
        x = torch.cat([x_src, x_dest], dim=0)
        edge_index[1] += n_src_nodes
        return x, edge_index

    def to(self, device):
        self.device = device
        return super(NodeAggregation, self).to(device)


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

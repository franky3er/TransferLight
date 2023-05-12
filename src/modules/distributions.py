import itertools

import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric import nn as pyg_nn

from src.modules.utils import FlexibleArgmax


class FlexibleCategorical(nn.Module):

    def __init__(self, logits: torch.Tensor, index: torch.Tensor):
        super(FlexibleCategorical, self).__init__()
        device = logits.get_device()
        self.device = "cpu" if device == -1 else f"cuda:{device}"
        if logits.dim() == 2 and index is None:
            logits, index = self._flatten(logits)
        self.logits = logits.type(torch.float64)
        self.index = index
        self.log_prob_fn = FlexibleCategoricalLogProb()
        self.log_prob_val = None
        self.sampling_fn = FlexibleCategoricalSampler()
        self.entropy_fn = FlexibleCategoricalEntropy()
        self.entropy_val = None

    def _flatten(self, logits: torch.Tensor):
        n_distributions = logits.size(0)
        n_categories = logits.size(1)
        index = torch.arange(0, n_distributions, device=self.device).unsqueeze(1).repeat(1, n_categories).view(-1)
        logits = logits.view(-1)
        return logits, index

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
        self.logits = self.logits.to(device)
        self.index = self.index.to(device)
        return super(FlexibleCategorical, self).to(device)


class FlexibleCategoricalLogProb(pyg_nn.MessagePassing):

    def __init__(self):
        super(FlexibleCategoricalLogProb, self).__init__(aggr="sum")

    def forward(self, logits: torch.Tensor, index: torch.LongTensor):
        logits = logits.unsqueeze(-1)
        edge_index = self._create_edge_index(index)
        exp_logits = torch.exp(logits)
        return self.propagate(edge_index, logits=logits, exp_logits=exp_logits).squeeze()

    def _create_edge_index(self, index: torch.Tensor):
        device = index.get_device()
        dist_logits = [[] for _ in range(torch.max(index) + 1)]
        for logit_idx, dist in enumerate(index):
            dist_logits[dist].append(logit_idx)
        edge_index = None
        for dist_logit_indices in dist_logits:
            src_nodes, dest_nodes = [], []
            for dest_node, src_node in itertools.product(dist_logit_indices, repeat=2):
                src_nodes.append(src_node)
                dest_nodes.append(dest_node)
            edge_index_dist = torch.tensor([src_nodes, dest_nodes], device=device)
            edge_index = edge_index_dist if edge_index is None else torch.cat([edge_index, edge_index_dist], dim=1)
        return edge_index

    @staticmethod
    def message(exp_logits_j: torch.Tensor) -> torch.Tensor:
        return exp_logits_j

    @staticmethod
    def update(aggregated_message: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return logits - torch.log(aggregated_message)


class FlexibleCategoricalEntropy(pyg_nn.MessagePassing):

    def __init__(self):
        super(FlexibleCategoricalEntropy, self).__init__(aggr="sum")

    def forward(self, logits: torch.Tensor, index: torch.Tensor):
        n_groups = torch.max(index) + 1
        edge_index = self._create_edge_index(index)
        probs = pyg.utils.softmax(logits, index).unsqueeze(1)
        log_probs = torch.log(probs)
        entropy = self.propagate(edge_index, probs=probs, log_probs=log_probs).squeeze()[:n_groups]
        return entropy

    def _create_edge_index(self, index: torch.Tensor):
        device = index.get_device()
        src_nodes = torch.arange(index.size(0), device=device).unsqueeze(0)
        dest_nodes = index.unsqueeze(0)
        edge_index = torch.cat([src_nodes, dest_nodes], dim=0)
        return edge_index

    def message(self, probs_j: torch.Tensor, log_probs_j: torch.Tensor):
        return - probs_j * log_probs_j


class FlexibleCategoricalSampler(nn.Module):

    def __init__(self):
        super(FlexibleCategoricalSampler, self).__init__()
        self.argmax = FlexibleArgmax()
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(loc=0, scale=1)
        self.device = "cpu"

    def forward(self, logits: torch.Tensor, index: torch.Tensor, return_sample_indices: bool = False):
        # Gumbel-max trick
        z = self.gumbel_dist.sample(logits.shape).to(logits.get_device())
        samples = self.argmax(logits + z, index, return_argmax_indices=return_sample_indices)
        return samples

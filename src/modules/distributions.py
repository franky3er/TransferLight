import torch
from torch import nn
from torch_geometric.utils import softmax

from src.modules.utils import group_argmax, group_max, group_sum, group_categorical_sample


class GroupCategorical(nn.Module):

    def __init__(self, logits: torch.Tensor = None, probs: torch.Tensor = None, index: torch.Tensor = None):
        super(GroupCategorical, self).__init__()
        assert (logits is not None) != (probs is not None)
        self.logits, self.probs = None, None
        if logits is not None:
            if logits.dim() == 2 and index is None:
                logits, index = self._flatten(logits)
            self.logits = normalize_logits(logits.type(torch.float64), index)
            device = logits.get_device()
        else:
            if probs.dim() == 2 and index is None:
                probs, index = self._flatten(logits)
            self.probs = probs
            device = probs.get_device()
        assert index is not None
        self.index = index
        self.device = "cpu" if device == -1 else f"cuda:{device}"
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(loc=0, scale=1)

    def _flatten(self, logits_or_probs: torch.Tensor):
        n_distributions = logits_or_probs.size(0)
        n_categories = logits_or_probs.size(1)
        index = torch.arange(0, n_distributions, device=self.device).unsqueeze(1).repeat(1, n_categories).view(-1)
        logits_or_probs = logits_or_probs.view(-1)
        return logits_or_probs, index

    def log_prob(self, values: torch.Tensor = None):
        log_probs = self.logits - torch.log(group_sum(torch.exp(self.logits), self.index))[self.index] \
            if self.logits is not None else torch.log(self.probs)
        return log_probs if values is None else log_probs[values]

    def entropy(self):
        probs = softmax(self.logits, self.index) if self.probs is None else self.probs
        log_probs = self.log_prob()
        return - group_sum(probs * log_probs, self.index)

    def sample(self, return_indices: bool = False):
        if self.logits is not None:
            # Gumbel-max trick
            device = self.logits.get_device()
            device = "cpu" if device == -1 else f"cuda:{device}"
            z = self.gumbel_dist.sample(self.logits.shape).to(device)
            return group_argmax(self.logits + z, self.index, return_indices=return_indices)
        return group_categorical_sample(self.probs, self.index, return_indices=return_indices)

    def to(self, device):
        self.logits = self.logits.to(device)
        self.index = self.index.to(device)
        return super(GroupCategorical, self).to(device)


def normalize_logits(logits: torch.Tensor, index: torch.LongTensor):
    logits = logits
    max_logits = group_max(logits, index)
    norm_logits = logits - max_logits[index]
    return norm_logits.squeeze()

from typing import List

import torch
from torch_geometric.utils import softmax
from torch_geometric.nn import aggr


def group_argmax(x: torch.Tensor, group_index: torch.Tensor, return_indices: bool = False, keepdim: bool = True):
    x = x.squeeze()
    assert x.size(0) == group_index.size(0)
    device = x.get_device()
    device = "cpu" if device == -1 else f"cuda:{device}"
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
    if return_indices:
        return argmax_values, argmax_indices
    return argmax_values.squeeze()


def group_categorical_sample(probs: torch.Tensor, group_index: torch.Tensor, return_indices: bool = False):
    probs = probs.squeeze()
    assert probs.size(0) == group_index.size(0)
    device = probs.get_device()
    device = "cpu" if device < 0 else device
    dtype = probs.dtype
    n_items = probs.size(0)
    n_groups = torch.max(group_index) + 1
    probs_repeated = probs.unsqueeze(1).repeat(1, n_groups)
    item_index = torch.arange(0, n_items, device=device)
    group_index = torch.nn.functional.one_hot(group_index)
    cumsum_group_index = torch.cumsum(group_index, dim=0) - 1
    dummy_probs = torch.zeros(n_items, n_groups, dtype=dtype, device=device)
    dummy_probs = (1 - group_index) * dummy_probs + group_index * probs_repeated
    dummy_probs = dummy_probs.T
    categorial = torch.distributions.Categorical(probs=dummy_probs)
    sample_index = categorial.sample().unsqueeze(0)
    sample_values = torch.gather(cumsum_group_index, 0, sample_index).squeeze()
    sample_indices = item_index[sample_index.squeeze()]
    return sample_values if not return_indices else sample_values, sample_indices


def group_sum(x: torch.Tensor, group_index: torch.LongTensor):
    assert x.size(0) == group_index.size(0)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    sum_aggr = aggr.SumAggregation()
    return sum_aggr(x, group_index).squeeze()


def group_max(x: torch.Tensor, group_index: torch.LongTensor):
    assert x.size(0) == group_index.size(0)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    max_aggr = aggr.MaxAggregation()
    return max_aggr(x, group_index).squeeze()


def group_mean(x: torch.Tensor, group_index: torch.LongTensor):
    assert x.size(0) == group_index.size(0)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    mean_aggr = aggr.MeanAggregation()
    return mean_aggr(x, group_index).squeeze()


def concat_features(features: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
    x = []
    for feature in features:
        if feature is not None:
            assert isinstance(feature, torch.Tensor)
            x.append(feature)
    return torch.cat(x, dim=dim)


def neighborhood_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, index: torch.LongTensor,
                           bias: torch.Tensor = 0.0) -> torch.Tensor:
    output_dim = v.size(1)
    hidden_dim = output_dim // heads
    q = q.view(-1, heads, hidden_dim)
    k = k.view(-1, heads, hidden_dim)
    v = v.view(-1, heads, hidden_dim)
    attention_coefficients = (1 / (hidden_dim ** (1/2))) * torch.sum(q * k, dim=-1) + bias
    attention_weights = softmax(attention_coefficients, index).view(-1, heads, 1)
    out = attention_weights * v
    return out.view(-1, output_dim).contiguous()


def sinusoidal_positional_encoding(pos: torch.LongTensor, dim: int, scalar: int = 10_000):
    assert dim % 2 == 0
    device = pos.get_device()
    device = "cpu" if device < 0 else device
    pos = pos.unsqueeze(-1).expand(-1, dim)
    k = (torch.arange(start=0, end=dim / 2, device=device).unsqueeze(-1) + 1).expand(-1, 2).reshape(1, -1)
    w_k = 1 / (scalar ** ((2 * k) / dim))
    i = torch.arange(start=0, end=dim, device=device).unsqueeze(0)
    even = i % 2 == 0
    uneven = i % 2 == 1
    return even * torch.sin(w_k * pos) + uneven * torch.cos(w_k * pos)

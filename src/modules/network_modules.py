from copy import copy
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj

from src.modules.base_modules import ResidualStack
from src.modules.message_passing import NeighborhoodAttention
from src.modules.utils import group_sum, group_mean


class TransferLightGraphEmbedding(nn.Module):

    def __init__(
            self,
            node_dims: Dict,
            edge_dims: Dict,
            pos: Dict,
            hidden_dim: int,
            node_updates: List,
            heads: int,
            n_residuals: int,
            dropout_prob: float
    ):
        super(TransferLightGraphEmbedding, self).__init__()
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.pos = pos
        self.hidden_dim = hidden_dim
        self.node_embedding_modules = nn.ModuleList()
        for dst_node, edges in node_updates:
            self.node_embedding_modules.append(
                TransferLightNodeEmbedding(copy(self.node_dims), copy(self.edge_dims), copy(self.pos),
                                           hidden_dim, dst_node, edges, heads, n_residuals, dropout_prob))
            self.node_dims[dst_node] = len(edges) * hidden_dim

    def forward(self, graph: HeteroData):
        for node_embedding_module in self.node_embedding_modules:
            graph = node_embedding_module(graph)
        return graph


class TransferLightNodeEmbedding(nn.Module):

    def __init__(
            self,
            node_dims: Dict,
            edge_dims: Dict,
            pos: Dict,
            hidden_dim: int,
            dst_node: str,
            edges: List[Tuple],
            heads: int,
            n_residuals: int,
            dropout_prob: float
    ):
        super(TransferLightNodeEmbedding, self).__init__()
        self.dst_node = dst_node
        self.partial_node_embedding_modules = nn.ModuleDict()
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.pos = pos
        for edge in edges:
            assert dst_node == edge[2]
            src_node = edge[0]
            src_dim = node_dims[src_node]
            dst_dim = node_dims[dst_node]
            edge_dim = edge_dims[edge]
            pos = "alibi" if self.pos[edge] else None
            self.partial_node_embedding_modules[";".join(edge)] = NeighborhoodAttention(
                src_dim, dst_dim, edge_dim, hidden_dim, heads, n_residuals, skip_connection=False,
                dropout_prob=dropout_prob, positional_encoding_method=pos)

    def forward(self, graph: HeteroData):
        node_embedding = None
        for edge, partial_node_embedding_module in self.partial_node_embedding_modules.items():
            edge = tuple(edge.split(";"))
            src_node, dst_node = edge[0], edge[2]
            x_src = graph[src_node].x if self.node_dims[src_node] > 0 else None
            x_dst = graph[dst_node].x if self.node_dims[dst_node] > 0 else None
            edge_attr = graph[edge[0], edge[1], edge[2]].edge_attr if self.edge_dims[edge] > 0 else None
            pos = graph[edge[0], edge[1], edge[2]].pos if self.pos[edge] else None
            edge_index = graph[edge[0], edge[1], edge[2]].edge_index
            partial_node_embedding = partial_node_embedding_module(x_src, x_dst, edge_attr, edge_index, pos)
            if node_embedding is None:
                node_embedding = partial_node_embedding
                continue
            node_embedding = torch.cat([node_embedding, partial_node_embedding], dim=-1)
        graph[self.dst_node].x = node_embedding
        return graph


class TransferLightQHead(nn.Module):

    def __init__(self, node_dims: Dict, edge_dims: Dict, hidden_dim: int):
        super(TransferLightQHead, self).__init__()
        self.linear_state_value = nn.Sequential(
            nn.Linear(node_dims["movement"], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.linear_action_advantage = nn.Sequential(
            nn.Linear(node_dims["phase"], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

    def forward(self, embedded_state: HeteroData):
        edge_index_movement_to_intersection = embedded_state["movement", "to", "intersection"].edge_index
        movement_embedding = embedded_state["movement"].x[edge_index_movement_to_intersection[0]]
        phase_embedding = embedded_state["phase"].x
        action_index = embedded_state["phase", "to", "intersection"].edge_index[1]
        state_values = group_sum(self.linear_state_value(movement_embedding), edge_index_movement_to_intersection[1])
        state_values = state_values[action_index]
        action_advantages = self.linear_state_value(phase_embedding).squeeze()
        action_advantages = action_advantages - group_mean(action_advantages, action_index)[action_index]
        action_values = state_values + action_advantages
        return action_values.squeeze(), action_index


class TransferLightActorCriticHead(nn.Module):

    def __init__(self, node_dims: Dict, edge_dims: Dict, hidden_dim: int):
        super(TransferLightActorCriticHead, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(node_dims["phase"], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(
            nn.Linear(node_dims["intersection"], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

    def forward(self, embedded_state: HeteroData):
        phase_embedding = embedded_state["phase"].x
        intersection_embedding = embedded_state["intersection"].x
        state_value = self.critic(intersection_embedding)
        action_index = embedded_state["phase", "to", "intersection"].edge_index[1]
        action_logits = self.actor(phase_embedding)
        return state_value.squeeze(), action_logits.squeeze(), action_index



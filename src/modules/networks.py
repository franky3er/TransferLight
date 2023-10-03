from collections import defaultdict
import sys
from typing import Dict, List

from torch import nn
from torch_geometric.data import HeteroData

from src.rl.problem_formulations import TransferLightProblemFormulation
from src.modules.message_passing import HeteroMessagePassing
from src.modules.utils import group_mean


class Network(nn.Module):

    @classmethod
    def create(cls, class_name: str, init_args: Dict):
        obj = getattr(sys.modules[__name__], class_name)(**init_args)
        assert isinstance(obj, Network)
        return obj


class TransferLightNetwork(Network):

    def __init__(self, network_type: str, hidden_dim: int = 64, n_attention_heads: int = 8,
                 dropout_prob: float = 0.1):
        super(TransferLightNetwork, self).__init__()
        metadata = TransferLightProblemFormulation.get_metadata()
        node_dims, edge_dims = metadata["node_dim"], metadata["edge_dim"]

        self.embed = TransferLightGraphEmbedding(node_dims, edge_dims, hidden_dim, n_attention_heads, dropout_prob)
        node_dims = {node: hidden_dim for node in node_dims.keys()}

        if network_type == "DQN":
            self.head = TransferLightQHead(node_dims, edge_dims, hidden_dim)
        elif network_type == "A2C":
            self.head = TransferLightActorCriticHead(node_dims, edge_dims, hidden_dim)
        else:
            raise Exception(f'network_type "{network_type}" not implemented')

    def forward(self, state: HeteroData) -> HeteroData:
        state = state.clone()
        embedded_state = self.embed(state)
        return self.head(embedded_state)


class TransferLightGraphEmbedding(nn.Module):

    def __init__(
            self,
            node_dims: Dict,
            edge_dims: Dict,
            hidden_dim: int,
            n_attention_heads: int = 8,
            dropout_prob: float = 0.0
    ):
        super(TransferLightGraphEmbedding, self).__init__()
        message_fct = {"class_name": "AttentionMessage",
                       "init_args": {"heads": n_attention_heads, "dropout_prob": dropout_prob}}
        aggregation_fct = "sum"
        update_fct = {"class_name": "StandardUpdate",
                      "init_args": {"dropout_prob": dropout_prob}}
        updates = [
            #[("segment", "to", "segment")],
            #[("segment", "to_down", "movement"), ("segment", "to_up", "movement")],
            #[("movement", "to", "movement")],
            [("movement", "to", "phase")],
            [("phase", "to", "phase")],
            [("phase", "to", "intersection")]
        ]
        self.layers = nn.ModuleList()
        for edge_types in updates:
            self.layers.append(TransferLightGraphEmbeddingLayer(edge_types, node_dims, edge_dims, hidden_dim,
                                                                message_fct, aggregation_fct, update_fct))
            updated_nodes = {node for _, _, node in edge_types}
            node_dims = {node: hidden_dim if node in updated_nodes else dim for node, dim in node_dims.items()}

    def forward(self, graph: HeteroData):
        for layer in self.layers:
            graph = layer(graph)
        return graph


class TransferLightGraphEmbeddingLayer(nn.Module):

    def __init__(
            self,
            edge_types: List,
            node_dims: Dict,
            edge_dims: Dict,
            hidden_dim: int,
            message_fct: Dict,
            aggregation_fct: str,
            update_fct: Dict
    ):
        super(TransferLightGraphEmbeddingLayer, self).__init__()
        node_edge_types = defaultdict(lambda: [])
        for edge_type in edge_types:
            dst_node = edge_type[2]
            node_edge_types[dst_node].append(edge_type)
        self.node_updates = nn.ModuleDict()
        for node, edge_types in node_edge_types.items():
            self.node_updates[node] = HeteroMessagePassing(edge_types, node_dims, edge_dims, hidden_dim, message_fct,
                                                           aggregation_fct, update_fct, skip_connection=True)

    def forward(self, graph: HeteroData):
        node_updates = {node: update(graph) for node, update in self.node_updates.items()}
        for node, update in node_updates.items():
            graph[node].x = update
        return graph


class TransferLightQHead(nn.Module):

    def __init__(self, node_dims: Dict, edge_dims: Dict, hidden_dim: int):
        super(TransferLightQHead, self).__init__()
        self.state_value_fct = nn.Sequential(
            nn.Linear(node_dims["intersection"], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.action_advantage_fct = nn.Sequential(
            nn.Linear(node_dims["phase"], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

    def forward(self, embedded_state: HeteroData):
        intersection_embedding = embedded_state["intersection"].x
        phase_embedding = embedded_state["phase"].x
        action_index = embedded_state["phase", "to", "intersection"].edge_index[1]
        state_values = self.state_value_fct(intersection_embedding).squeeze()
        state_values = state_values[action_index]
        action_advantages = self.action_advantage_fct(phase_embedding).squeeze()
        action_advantages = action_advantages - group_mean(action_advantages, action_index)[action_index]
        action_values = state_values + action_advantages
        return action_values, action_index


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

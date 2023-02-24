from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.init import orthogonal_
from torch_geometric import nn as pyg_nn

from src.models.dqn_heads import DuelingHead
from src.models.modules import HeteroModule, PhaseDemandLayer, PhaseCompetitionLayer


class DQN(nn.Module):

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.1)


class HieraGLightDQN(DQN):

    def __init__(self, movement_dim: int, phase_dim: int, hidden_dim: int):
        super(HieraGLightDQN, self).__init__()
        self.movement_demand_layer = HeteroModule({
            "movement": nn.Sequential(
                nn.Linear(in_features=movement_dim, out_features=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.ReLU(inplace=True)
            )
        })

        self.phase_demand_layer = pyg_nn.HeteroConv({
            ("movement", "to", "phase"): PhaseDemandLayer()
        })
        self.phase_competition_layer = PhaseCompetitionLayer(hidden_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        x_dict = self.movement_demand_layer(x_dict)
        x_dict.update(self.phase_demand_layer(x_dict, edge_index_dict))
        out = self.phase_competition_layer(x_dict["phase"], edge_index_dict[("phase", "to", "phase")])
        print(out.shape)
        # TODO: phase-to-phase edge index not corrent yet


class LitDQN(DQN):

    def __init__(self, state_size: int, hidden_size: int, action_size: int):
        super(LitDQN, self).__init__()
        self.state_embedding = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.dqn_head = DuelingHead(hidden_size, hidden_size, action_size)
        self.apply(self._init_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.state_embedding(x)
        return self.dqn_head(x)

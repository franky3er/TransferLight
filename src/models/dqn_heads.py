import torch
from torch import nn


class DuelingHead(nn.Module):

    def __init__(self, in_size: int, hidden_size: int, action_size: int):
        super(DuelingHead, self).__init__()
        self.state_value_stream = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=hidden_size),
            nn.PReLU(),
            nn.Linear(in_features=in_size, out_features=1)
        )
        self.action_advantage_stream = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=hidden_size),
            nn.PReLU(),
            nn.Linear(in_features=in_size, out_features=action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state_values = self.state_value_stream(x)
        action_advantages = self.action_advantage_stream(x)
        action_values = state_values + (action_advantages - torch.mean(action_advantages, dim=-1, keepdim=True))
        return action_values

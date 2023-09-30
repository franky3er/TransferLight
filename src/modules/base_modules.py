import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, last_activation: bool = True, dropout_prob: float = 0.0):
        super(ResidualBlock, self).__init__()
        self.identity = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim, bias=False)
        self.residual_function = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.last_activation = last_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual_function(x) + self.identity(x)
        return torch.relu(out) if self.last_activation else out


class ResidualStack(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, stack_size: int, last_activation: bool = True,
                 dropout_prob: float = 0.0):
        super(ResidualStack, self).__init__()
        residual_blocks = []
        for i in range(stack_size - 1):
            residual_blocks.append(ResidualBlock(input_dim if i == 0 else output_dim, output_dim,
                                                 dropout_prob=dropout_prob))
        residual_blocks.append(ResidualBlock(input_dim if stack_size == 1 else output_dim, output_dim, last_activation,
                                             dropout_prob=dropout_prob))
        self.residual_blocks = nn.Sequential(*tuple(residual_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_blocks(x)

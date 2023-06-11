import torch
from torch import nn


class MultiInputSequential(nn.Sequential):

    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


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

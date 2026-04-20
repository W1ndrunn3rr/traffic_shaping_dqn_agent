from torchrl.modules import NoisyLinear
from torch import nn
import torch


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, num_actions, dense_size):
        super().__init__()
        self.value = nn.Sequential(
            NoisyLinear(state_dim, dense_size),
            nn.ReLU(),
            NoisyLinear(dense_size, dense_size),
            nn.ReLU(),
            NoisyLinear(dense_size, 1),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(state_dim, dense_size),
            nn.ReLU(),
            NoisyLinear(dense_size, dense_size),
            nn.ReLU(),
            NoisyLinear(dense_size, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.value(x)
        adv = self.advantage(x)
        return val + (adv - adv.mean(dim=-1, keepdim=True))

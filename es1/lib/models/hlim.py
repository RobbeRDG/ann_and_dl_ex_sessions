import torch
from torch import nn


class HardLimit(nn.Module):
    values = torch.tensor([0.])

    def forward(self, x):
        return torch.heaviside(x, self.values)
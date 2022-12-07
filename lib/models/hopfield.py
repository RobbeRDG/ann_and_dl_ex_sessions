import torch
from torch import nn
from torch.nn.functional import hardtanh


class Hopfield:
    def __init__(self, attractors: torch.Tensor):
        """
        Create a Hopfield network zith N neurons.

        Args:
            attractors (torch.Tensor): tensor of length L containing
                L attractors with components equal to ±1.
        """
        self.shape = attractors.shape[1:]
        self.attractors = attractors
        attractors = attractors.flatten(start_dim=1)
        L, N = attractors.shape

        attractors = attractors[..., None]

        self.w = (attractors @ attractors.transpose(-1, -2)).sum(dim=0) / N
        self.w.requires_grad = False

    def __call__(self, x: torch.Tensor, sync=True):
        """
        Args:
            x (torch.Tensor): Batch of patterns.
            num_step (int): The number of steps.
        """
        x = x.flatten(start_dim=1)

        out = hardtanh(self.w @ x.T).T

        if sync:
            x = out
        else:
            idxs = torch.randint(0, x.shape[1],
                                 (x.shape[0],))
            rng = torch.arange(x.shape[0])
            x[rng, idxs] = out[rng, idxs]

        x = out.unflatten(1, self.shape)

        return x
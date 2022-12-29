import torch
from torch import nn
from torch.nn.functional import hardtanh


class Hopfield:
    def __init__(self, attractors: torch.Tensor, tau=0.1):
        """
        Create a Hopfield network zith N neurons.

        Args:
            attractors (torch.Tensor): tensor of length L containing
                L attractors with components equal to ±1.
        """
        self.shape = attractors.shape[1:]
        self.attractors = attractors
        attractors = attractors.flatten(start_dim=1)

        # 0. Compose n x m matrix of alphas
        A = attractors.T
        a_m = A[:, -1:]
        n, m = A.shape

        # 1. Compute n x (m-1) matrix Y
        Y = A[:, :-1] - a_m

        # 2. Perform SVD of Y to get U, V and S
        U, S, Vh = torch.linalg.svd(Y)
        k = (S != 0).sum()

        # 3. Compute T_plus and T_min
        T_plus = U[:, :k] @ U[:, :k].T
        T_min  = U[:, k:] @ U[:, k:].T

        # 4. Compute T_tau
        T_tau = T_plus - tau*T_min

        # 5. Compute I_tau
        I_tau = a_m - T_tau @ a_m

        # Set weights and bias
        self.w = T_tau
        self.b = I_tau

    def __call__(self, x: torch.Tensor, sync=True):
        """
        Args:
            x (torch.Tensor): Batch of patterns.
            num_step (int): The number of steps.
        """
        x = x.flatten(start_dim=1)

        out = hardtanh(x.T + self.w @ x.T + self.b).T

        if sync:
            x = out
        else:
            idxs = torch.randint(0, x.shape[1],
                                 (x.shape[0],))
            rng = torch.arange(x.shape[0])
            x[rng, idxs] = out[rng, idxs]

        x = out.unflatten(1, self.shape)

        return x
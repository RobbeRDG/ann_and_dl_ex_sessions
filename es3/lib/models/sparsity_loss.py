import torch


def sparsity_loss_fn(z, rho):
    # Compute the mean encoding
    z_mean = z.mean(dim=0)

    # Compute KL divergence
    dkl = (
        - rho * torch.log(z_mean)
        - (1 - rho)*torch.log(1 - z_mean)
    )

    return dkl.mean()
import torch
from torch import tensor

from ..utils.generate import rand_mult_norm


def get_corr_samples(n_samples):
    eig_vals = tensor([0.01, 1.])
    eig_vecs = tensor([
        [1., 1],
        [-1, 1]
    ]).float()
    mu = tensor([100., 10])
    samples = rand_mult_norm(n_samples, mu, eig_vals, eig_vecs, seed=42)
    return samples


def get_uncorr_samples(n_samples):
    eig_vals = torch.ones(2)
    eig_vecs = torch.eye(2)
    mu = tensor([100., 10])
    samples = rand_mult_norm(n_samples, mu, eig_vals, eig_vecs, seed=42)
    return samples
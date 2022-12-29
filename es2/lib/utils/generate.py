import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def rand_mult_norm(n_samples, mu, sigma_eig_vals, sigma_eig_vecs, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    sigma_eig_vals = torch.diag(sigma_eig_vals)
    sigma = sigma_eig_vecs @ sigma_eig_vals @ sigma_eig_vecs.inverse()
    distr = MultivariateNormal(mu, sigma)
    return distr.sample((n_samples,))
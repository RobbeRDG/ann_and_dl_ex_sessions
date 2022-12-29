import matplotlib.pyplot as plt
import numpy as np
import torch


def check_ndim(ndim):
    if ndim > 3:
        raise ValueError(f'Cannot plot more than 3 dims, got {ndim}')


def get_ndim_plot(ndim):
    check_ndim(ndim)
    fig, ax = plt.subplots(
        subplot_kw=dict(
            projection="3d" if ndim == 3 else None,
        ),
    )
    ax.computed_zorder = False
    return fig, ax


def plot_trajectory(trajectory, ax=None):
    ndim = trajectory[0].shape[1]

    if ax is None:
        _, ax = get_ndim_plot(ndim)

    plot_list = [
        np.array(
            torch.stack([p[:, i] for p in trajectory],
                        dim=1)
        )
        for i in range(ndim)
    ]

    for coords in zip(*plot_list):
        ax.plot(*coords, marker='.', alpha=0.5, zorder=0)

    return ax


def plot_attractors(attractors, ax=None, color='green'):
    ndim = attractors.shape[1]

    if ax is None:
        _, ax = get_ndim_plot(ndim)

    ax.scatter(*np.array(attractors).T, marker='*',
               s=72, fc=color, zorder=1)
    return ax


def get_hopfield_evolution(net, init_points, num_steps=20):
    trajectory = [init_points]

    for _ in range(num_steps):
        init_points = net(init_points)
        trajectory.append(init_points)

    return trajectory


def plot_hopfield_evolution(net, init_points, num_steps=20):
    trajectory = get_hopfield_evolution(net, init_points, num_steps)
    ax = plot_trajectory(trajectory)
    plot_attractors(net.attractors, ax)
    return ax


def converged_to_attractor(outputs, attractors):
    return (outputs[:, None, ...] == attractors).all(dim=-1).any(dim=-1)
import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objs as go


def check_ndim(ndim):
    if ndim > 3:
        raise ValueError(f'Cannot plot more than 3 dims, got {ndim}')


def plot_trajectory(trajectory, ax=None):
    ndim = trajectory[0].shape[1]

    trajectory = torch.stack(trajectory).permute(2, 1, 0)

    if ndim == 2:
        fig = go.Figure(data=[
            go.Scatter(x=x, y=y, mode='lines+markers')
            for x, y in zip(*trajectory)
        ])

    elif ndim == 3:
        fig = go.Figure(data=[
            go.Scatter3d(x=x, y=y, z=z, marker=dict(size=3))
            for x, y, z in zip(*trajectory)
        ])

    return fig


def plot_attractors(attractors, fig=None, color='green'):
    ndim = attractors.shape[1]

    if fig is None:
        fig = go.Figure()

    for i, attractor in enumerate(attractors):
        if ndim == 2:
            fig.add_scatter(
                x=[attractor[0]], y=[attractor[1]], marker_symbol='cross',
                mode="markers", marker_size=15,
                name=f'Attractor {i + 1}'
            )
        elif ndim == 3:
            fig.add_scatter3d(
                x=[attractor[0]], y=[attractor[1]], z=[attractor[2]],
                marker_symbol='cross', mode="markers", marker_size=15,
                name=f'Attractor {i + 1}'
            )

    fig.update_xaxes(
        scaleanchor = "y",
        scaleratio = 1,
    )

    return fig


def get_hopfield_evolution(net, init_points, num_steps=20):
    trajectory = [init_points]

    for _ in range(num_steps):
        init_points = net(init_points)
        trajectory.append(init_points)

    return trajectory


def plot_hopfield_evolution(net, init_points, num_steps=20):
    trajectory = get_hopfield_evolution(net, init_points, num_steps)
    fig = plot_trajectory(trajectory)
    plot_attractors(net.attractors, fig)
    return fig


def converged_to_attractor(outputs, attractors):
    return (outputs[:, None, ...] == attractors).all(dim=-1).any(dim=-1)
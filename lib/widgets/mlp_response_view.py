from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import plotly.graph_objs as go

from ..models.mlp import MLP
from .mlp_view import MLPView


class MLPResponseView(MLPView):
    def __init__(
        self,
        mlp: MLP,
        data_x: torch.Tensor = None,
        data_y: torch.Tensor = None,
        top=None,
        right=None,
        bottom=None,
        left=None,
        samples_per_dim=1000,
        tolerance=0.,  # Margin before considering a prediction correct
        proj_3d=False,
    ):
        self.samples_per_dim = samples_per_dim
        self._top = top
        self._right = right
        self._bottom = bottom
        self._left = left
        self._auto_top = None
        self._auto_right = None
        self._auto_bottom = None
        self._auto_left = None

        super().__init__(mlp, data_x, data_y)

        plt.ioff()
        if proj_3d:
            self.fig, _ = plt.subplots(subplot_kw={'projection': '3d'})
        else:
            self.fig = plt.figure()
        plt.ion()
        self.widget = self.fig.canvas
        self.ax_im = None
        self.ax_plt = None
        self.ax_sc = None
        self.ax_surf = None
        self.cb = None
        self.x0_data_plot = []
        self.x1_data_plot = []
        self.y_data_plot = []
        self.scatter_edge_colors = []
        self.cmap = cm.viridis
        self.tolerance = tolerance
        self.proj_3d = proj_3d

        self.render()
        plt.close()

    @property
    def data_x(self):
        return self._data_x

    @property
    def data_y(self):
        return self._data_y

    @property
    def top(self):
        return self._top or self._auto_top or 10

    @property
    def bottom(self):
        return self._bottom or self._auto_bottom or -10

    @property
    def right(self):
        return self._right or self._auto_right or 10

    @property
    def left(self):
        return self._left or self._auto_left or -10

    def _get_axis_bounds(self, data, rel_margin=0.05):
        data_width = (data.max() - data.min()).abs()
        margin = rel_margin * data_width
        min_bound = (data.min() - margin).floor()
        max_bound = (data.max() + margin).ceil()
        return min_bound, max_bound

    @data_x.setter
    def data_x(self, val):
        self._data_x = val
        if val is None:
            return

        self._auto_left, self._auto_right = self._get_axis_bounds(val[:, 0])

        if val.shape[1] == 2:
            # data_x[:, 1] on y
            self._auto_bottom, self._auto_top = self._get_axis_bounds(
                val[:, 1]
            )

    @data_y.setter
    def data_y(self, val):
        self._data_y = val
        if val is None:
            return

        if self.data_x.shape[1] == 1:
            # data_y on y
            self._auto_bottom, self._auto_top = self._get_axis_bounds(val)

    @top.setter
    def top(self, val):
        self._top = val

    @bottom.setter
    def bottom(self, val):
        self._bottom = val

    @right.setter
    def right(self, val):
        self._right = val

    @left.setter
    def left(self, val):
        self._left = val

    def render(self):
        plt.ion()
        if self.mlp.num_input_neurons == 1:
            x, y = get_mlp_response_1d(
                self.mlp,
                right=self.right, left=self.left,
                samples_per_dim=self.samples_per_dim,
            )

            self.update_scatter_data_1d()

            ax = plt.gca()
            if self.ax_plt is None:
                self.ax_plt = plt.plot(x, y)
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                ax.set_aspect('equal')
                self.ax_sc = ax.scatter(
                    self.x0_data_plot,
                    self.y_data_plot,
                )
            else:
                self.ax_plt[0].set_data(x, y)
                self.ax_sc.set_offsets(np.c_[self.x0_data_plot,
                                             self.y_data_plot])
            self.update_xylim(ax)
        elif self.mlp.num_input_neurons == 2 and self.proj_3d:
            x1_mesh, x2_mesh, y_mesh, fills = get_mlp_response_2d(
                self.mlp, self.top, self.right,
                self.bottom, self.left,
                samples_per_dim=self.samples_per_dim,
            )
            norm = colors.Normalize(vmin=y_mesh.min(), vmax=y_mesh.max())

            x1_mesh = x1_mesh.detach().cpu().numpy()
            x2_mesh = x2_mesh.detach().cpu().numpy()
            y_mesh = norm(y_mesh.detach().cpu().numpy())

            ax = plt.gca()
            if self.ax_surf is None:
                self.ax_surf = ax.plot_surface(x1_mesh, x2_mesh, y_mesh,
                                               cmap=self.cmap)
                ax.set_xlabel('$x_0$')
                ax.set_ylabel('$x_1$')
                ax.set_zlabel('$y$')
            else:
                self.ax_surf.remove()
                self.ax_surf = ax.plot_surface(x1_mesh, x2_mesh, y_mesh,
                                               cmap=self.cmap)

            self.update_xylim(ax)
            self.update_scatter_data_2d()
            ax.zaxis.set_major_formatter(
                lambda z, pos: f'{norm.inverse(z):.1f}'
            )
            self.ax_sc = ax.scatter3D(
                self.x0_data_plot,
                self.x1_data_plot,
                norm(self.y_data_plot),
                c=norm(self.y_data_plot),
                ec=self.scatter_edge_colors,
                cmap=self.cmap,
            )
        elif self.mlp.num_input_neurons == 2 and not self.proj_3d:
            _, _, _, img = get_mlp_response_2d(
                self.mlp, self.top, self.right,
                self.bottom, self.left,
                samples_per_dim=self.samples_per_dim
            )
            norm = colors.Normalize(vmin=img.min(), vmax=img.max())

            img = norm(img.detach().cpu().numpy())

            ax = plt.gca()
            if self.ax_im is None:
                self.ax_im = plt.imshow(img)
                ax.set_xlabel('$x_0$')
                ax.set_ylabel('$x_1$')
            else:
                self.ax_im.set_data(img)

            if self.ax_sc is not None:
                self.ax_sc.remove()

            self.update_scatter_data_2d()
            self.ax_sc = ax.scatter(
                self.x0_data_plot,
                self.x1_data_plot,
                c=self.y_data_plot,
                ec=self.scatter_edge_colors,
                cmap=self.cmap,
            )

            self.update_colorbar(self.ax_im, norm)
            self.ax_im.set_extent([self.left, self.right,
                                   self.bottom, self.top])
        else:
            raise ValueError('Number of input neurons must be '
                             '1 or 2 for visualization')

    def update_colorbar(self, ax, norm):
        if self.cb is not None:
            self.cb.remove()

        cb_ticks = [i/5 for i in range(6)]
        self.cb = self.fig.colorbar(ax, ticks=cb_ticks)
        self.cb.set_label("$y$")

        new_ticklabels = [
            f'{norm.inverse(t):.1f}'
            for t in cb_ticks
        ]
        self.cb.ax.set_yticklabels(new_ticklabels)

    def update_xylim(self, ax):
        ax.set_xlim(self.left, self.right)
        ax.set_ylim(self.bottom, self.top)

    def update_scatter_data_2d(self):
        if self.data_x is None or self.data_y is None:
            return

        self.x0_data_plot.clear()
        self.x1_data_plot.clear()
        self.y_data_plot.clear()
        self.scatter_edge_colors.clear()

        y_pred = self.mlp(self.data_x)
        for y_t, y_p in zip(self.data_y.flatten(), y_pred.flatten()):
            self.scatter_edge_colors.append(
                'green'
                if (y_t - self.tolerance) <= y_p <= (y_t + self.tolerance)
                else 'red'
            )
            self.y_data_plot.append(y_t.item())

        for x0, x1 in self.data_x:
            self.x0_data_plot.append(x0.item())
            self.x1_data_plot.append(x1.item())

    def update_scatter_data_1d(self):
        if self.data_x is None or self.data_y is None:
            return

        self.x0_data_plot.clear()
        self.y_data_plot.clear()

        for y_t in self.data_y.flatten():
            self.y_data_plot.append(y_t.item())

        for x0 in self.data_x.flatten():
            self.x0_data_plot.append(x0.item())


def get_mlp_response_2d(
    mlp: nn.Module,
    top, right,
    bottom, left,
    samples_per_dim,
):
    x1s = torch.linspace(left, right, samples_per_dim)
    x2s = torch.linspace(bottom, top, samples_per_dim)

    x1_mesh, x2_mesh = torch.meshgrid(x1s, x2s, indexing="ij")

    samples = torch.hstack([x1_mesh.flatten()[:, None],
                            x2_mesh.flatten()[:, None]])

    with torch.no_grad():
        out = mlp(samples)

    out = out.reshape(samples_per_dim, samples_per_dim)

    fills = out.rot90(1, [0, -1])

    return x1_mesh, x2_mesh, out, fills


def get_mlp_response_1d(
    mlp: nn.Module,
    right, left,
    samples_per_dim,
):
    samples = torch.linspace(left, right, samples_per_dim)[..., None]

    with torch.no_grad():
        out = mlp(samples)

    heights = out.reshape(samples_per_dim)
    return samples, heights


def show_mlp_response_3d(
    mlp, data_x, data_y,
    top=5, right=5, bottom=-5, left=-5,
    samples_per_dim=100
):
    # Get MLP response
    x1_mesh, x2_mesh, y_mesh, fills = get_mlp_response_2d(
        mlp, top=top, right=right, bottom=bottom, left=left,
        samples_per_dim=samples_per_dim
    )

    z = data_y[:, 0]

    fig = go.Figure(data=[
        # Create surface plot of MLP respone
        go.Surface(x=x1_mesh[:, 0], y=x2_mesh[0, :], z=y_mesh.T,
                   colorscale='Viridis', opacity=0.7),

        # Add ground truth data
        go.Scatter3d(
            x=data_x[:, 0], y=data_x[:, 1], z=z,
            mode='markers',
            marker=dict(
                color=z,
                colorscale='Viridis',
            )
        )
    ])

    return fig.show()

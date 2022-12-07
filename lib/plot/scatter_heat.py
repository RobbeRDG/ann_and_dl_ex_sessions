import matplotlib.pyplot as plt
from matplotlib import cm
import torch


def scatter_heat(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """
    Scatter plot the x and y values, color coding with the respective z values.
    """
    cmap = cm.get_cmap('viridis')

    fix, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, cmap=cmap)
    plt.colorbar(sc)
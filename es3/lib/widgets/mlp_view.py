from ..models.mlp import MLP
import torch


class MLPView:
    def __init__(
        self,
        mlp: MLP,
        data_x: torch.Tensor=None,
        data_y: torch.Tensor=None,
    ):
        self.mlp = mlp
        self.widget = None
        self.data_x = data_x
        self.data_y = data_y

    def render(self):
        raise NotImplementedError
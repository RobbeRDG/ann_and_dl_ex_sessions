from typing import List

import torch
from torch import nn

from .hlim import HardLimit


ACTIV_FUNCS = ['Tanh', 'ReLU', 'Sigmoid', 'Identity', 'HardLimit']


class MLP(nn.Module):
    def __init__(
        self,
        num_input_neurons=1,
        num_output_neurons=1,
        num_hidden_neurons=[],
        activation=None,
        output_activation=None,
    ):
        super().__init__()
        self._num_input_neurons = num_input_neurons
        self._num_output_neurons = num_output_neurons
        self._num_hidden_neurons = num_hidden_neurons
        self._activation = activation
        self._output_activation = output_activation
        self._model = None
        self._weights = None
        self._biases = None
        self.model

    def forward(self, x):
        model = self.model
        return model(x.float())

    @property
    def num_input_neurons(self):
        return self._num_input_neurons

    @property
    def num_output_neurons(self):
        return self._num_output_neurons

    @property
    def num_hidden_neurons(self):
        return self._num_hidden_neurons

    @property
    def activation(self):
        return self._activation

    @property
    def output_activation(self):
        return self._output_activation

    @property
    def model(self):
        if self._model is None:
            self._model = _create_mlp(
                num_input_neurons=self.num_input_neurons,
                num_output_neurons=self.num_output_neurons,
                num_hidden_neurons=self.num_hidden_neurons,
                activation=self.activation,
                output_activation=self.output_activation,
            )
            if self._weights is not None:
                self.weights = self._weights
            if self._biases is not None:
                self.biases = self._biases

        return self._model

    @property
    def weights(self):
        if self._weights is None:
            self._weights = get_mlp_weights(self.model)
        return self._weights

    @property
    def biases(self):
        if self._biases is None:
            self._biases = get_mlp_biases(self.model)
        return self._biases

    def set_weight(self, layer, input_neuron, output_neuron, new_weight):
         with torch.no_grad():
            self.weights[layer][output_neuron, input_neuron] = new_weight

    def set_bias(self, layer, output_neuron, new_bias):
        with torch.no_grad():
            self.biases[layer][output_neuron] = new_bias

    @num_input_neurons.setter
    def num_input_neurons(self, val):
        self.set_mlp_prop("_num_input_neurons", val)
        self._weights = None
        self._biases = None

    @num_output_neurons.setter
    def num_output_neurons(self, val):
        self.set_mlp_prop("_num_output_neurons", val)
        self._weights = None
        self._biases = None

    @num_hidden_neurons.setter
    def num_hidden_neurons(self, val):
        self.set_mlp_prop("_num_hidden_neurons", val)
        self._weights = None
        self._biases = None

    @activation.setter
    def activation(self, val):
        self.set_mlp_prop("_activation", val)

    @output_activation.setter
    def output_activation(self, val):
        self.set_mlp_prop("_output_activation", val)

    def set_mlp_prop(self, prop, val):
        if not getattr(self, prop) == val:
            setattr(self, prop, val)
            self._model = None

    @weights.setter
    def weights(self, val):
        set_mlp_weights(self._model, val)
        self._weights = None

    @biases.setter
    def biases(self, val):
        set_mlp_biases(self._model, val)
        self._biases = None


def get_activation(activation):
    if activation is None:
        activation = 'Identity'
    if activation == 'HardLimit':
        return HardLimit
    elif not activation in nn.__dict__:
        raise ValueError(f'Unknown activation "{activation}"')
    else:
        return nn.__dict__[activation]


def _create_mlp(
    num_input_neurons: int,
    num_output_neurons: int,
    num_hidden_neurons: List[int] = [],
    activation: str = 'Tanh',
    output_activation: str = 'Identity',
):
    NonLinAct = get_activation(activation)
    OutAct = get_activation(output_activation)

    neurons_per_layer = [
        num_input_neurons,
        *num_hidden_neurons,
        num_output_neurons
    ]

    children = []

    for num_in, num_out in zip(neurons_per_layer,
                               neurons_per_layer[1:]):
        children.extend([
            nn.Linear(num_in, num_out),
            NonLinAct()
        ])

    children[-1] = OutAct()

    return nn.Sequential(*children)


def get_mlp_weights(mlp: nn.Module):
    return [layer.weight for layer in mlp if isinstance(layer, nn.Linear)]


def get_mlp_biases(mlp: nn.Module):
    return [layer.bias for layer in mlp if isinstance(layer, nn.Linear)]


def set_mlp_weights(mlp, new_weights):
    old_weights = get_mlp_weights(mlp)

    with torch.no_grad():
        for old_weight, new_weight in zip(old_weights, new_weights):
            old_weight.copy_(new_weight)


def set_mlp_biases(mlp, new_biases):
    old_biases = get_mlp_biases(mlp)

    with torch.no_grad():
        for old_bias, new_bias in zip(old_biases, new_biases):
            old_bias.copy_(new_bias)

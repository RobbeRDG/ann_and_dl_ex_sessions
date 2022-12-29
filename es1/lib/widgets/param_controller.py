from functools import partial

import ipywidgets as widgets
import numpy as np

from .controller import MLPController
from ..models.perceptron import Perceptron


class MLPParamController(MLPController):
    def get_controller_widget(self):
        parent_widget = super().get_controller_widget()

        return widgets.HBox([
            parent_widget, self.param_widgets
        ])

    def create_controller_widgets(self):
        super().create_controller_widgets()
        self.create_param_widgets()

    def create_param_widgets(self):
        param_widgets = []
        weight_widgets = [
            np.empty(layer_weights.shape, dtype=object)
            for layer_weights in self.mlp.weights
        ]
        bias_widgets = [
            np.empty(layer_biases.shape, dtype=object)
            for layer_biases in self.mlp.biases
        ]

        for i, (layer_weights, layer_biases) in enumerate(zip(self.mlp.weights,
                                                              self.mlp.biases)):
            num_output, num_input = layer_weights.shape
            assert len(layer_biases) == num_output

            layer_widgets = [
                widgets.Label(f'Layer {i} -> {i+1}')
            ]
            layer_weight_widgets = []
            layer_bias_widgets = []

            for k in range(num_output):
                for j in range(num_input):
                    weight_widget = widgets.FloatSlider(
                        description=f'W[{j},{k}]',
                        value=layer_weights[k, j],
                        min=-3,
                        max=3,
                        readout_format='.2f',
                        step=0.01,
                    )
                    weight_widget.observe(
                        partial(self.on_weight_change, i, j, k),
                        names='value'
                    )
                    layer_widgets.append(weight_widget)
                    weight_widgets[i][k, j] = weight_widget

                if layer_biases[k] is not None:
                    bias_widget = widgets.FloatSlider(
                        description=f'B[{k}]',
                        value=layer_biases[k],
                        min=-3,
                        max=3,
                        readout_format='.2f',
                        step=0.01,
                    )
                    bias_widget.observe(
                        partial(self.on_bias_change, i, k),
                        names='value'
                    )
                    layer_widgets.append(bias_widget)
                    bias_widgets[i][k] = bias_widget

            layer_widgets_box = widgets.VBox(layer_widgets)
            layer_widgets_box.layout.align_items = 'center'
            param_widgets.append(layer_widgets_box)

        self.weight_widgets = weight_widgets
        self.bias_widgets = bias_widgets
        self.param_widgets = widgets.HBox(param_widgets)

    def on_weight_change(
        self, layer, input_neuron, output_neuron,
        change
    ):
        self.mlp.set_weight(layer, input_neuron, output_neuron, change['new'])
        self.render_views()

    def on_bias_change(
        self, layer, output_neuron,
        change
    ):
        self.mlp.set_bias(layer, output_neuron, change['new'])
        self.render_views()


class PerceptronParamController(MLPParamController):
    def __init__(
        self,
        perceptron=None,
        views=None,
        num_input_neurons=None,
        control_num_inputs=False,
        proj_3d=False,
    ):
        mlp = (
            Perceptron(num_input_neurons=num_input_neurons)
            if perceptron is None
            else perceptron
        )
        super().__init__(
            mlp=mlp,
            views=views,
            control_num_inputs=control_num_inputs,
            control_activation=False,
            control_hlayers=False,
            proj_3d=proj_3d
        )

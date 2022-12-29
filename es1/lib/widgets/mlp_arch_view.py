import torch

from ..models.mlp import MLP
from .mlp_view import MLPView


class MLPArchView(MLPView):
    def __init__(
        self,
        mlp: MLP,
        data_x: torch.Tensor = None,
        data_y: torch.Tensor = None,
        cr=50,
        fill_input='#F0F0F0',
        fill_output='#F0F0F0',
        fill_hidden='#FFB700',
        fill_bias='#E0E0E0',
        connect_stroke='#555555',
        h_pad=50,
        v_pad=25,
    ):
        super().__init__(mlp, data_x, data_y)
        self.cr = cr
        self.fill_input = fill_input
        self.fill_output = fill_output
        self.fill_hidden = fill_hidden
        self.fill_bias = fill_bias
        self.connect_stroke = connect_stroke
        self.h_pad = h_pad
        self.v_pad = v_pad
        self.svg_data = ''

    def _repr_html_(self):
        mlp_svg, mlp_width, mlp_height = draw_mlp_from_params(
            self.mlp.weights, self.mlp.biases,
            cr=self.cr, fill_input=self.fill_input,
            fill_output=self.fill_output, fill_hidden=self.fill_hidden,
            fill_bias=self.fill_bias, connect_stroke=self.connect_stroke,
            h_pad=self.h_pad, v_pad=self.v_pad
        )
        svg = f'<svg width="100%" height="100%" viewbox="0 0 {mlp_width} {mlp_height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="svg-content">'
        svg += mlp_svg
        svg += '</svg>'

        html = f"""
        <!DOCTYPE html>
        <html>
          <head>
            <style>
              .svg-container {{
                max-width: 300px;
              }}
              .svg-content {{
                max-width: 100%;
              }}
            </style>
          </head>
          <body>
            <div class="svg-container">
            { svg }
            </div>
          </body>
        <html>
        """
        return html


def get_neuron_centers(layers, cr, h_pad, v_pad):
    c_size = 2*cr
    layer_heights = [
        (num_neur * c_size) + (num_neur - 1) * v_pad
        for num_neur in layers
    ]

    svg_width = c_size + (h_pad + c_size)*len(layers)
    svg_height = max(layer_heights)

    return [
        [((c_size + h_pad)*i + cr,
          (svg_height - layer_height)/2 + (c_size + v_pad)*j + cr)
         for j in range(num_neur)]
        for i, (num_neur, layer_height) in enumerate(zip(layers,
                                                         layer_heights))
    ], svg_width, svg_height


def is_hidden(i, centers):
    return not (is_input(i, centers) or is_output(i, centers))


def is_input(i, centers):
    return i == 0


def is_output(i, centers):
    return i == len(centers) - 1


def is_bias_neuron(i, j, weights, biases):
    return (
        layer_has_bias_neuron(i, biases)
        and j == weights[i].shape[1]
    )


def param_to_width(param):
    return abs(param) * 10


def draw_neurons(
    centers, cr,
    weights,
    biases=None,
    fill_input='green',
    fill_output='blue',
    fill_hidden='orange',
    fill_bias='lightgray',
):
    svg = ''
    for i, layer_centers in enumerate(centers):
        has_bias = layer_has_bias_neuron(i, biases)
        if i == 0:
            fill = fill_input
        elif i == len(centers) - 1:
            fill = fill_output
        else:
            fill = fill_hidden

        for j, (cx, cy) in enumerate(layer_centers):
            if has_bias and is_bias_neuron(i, j, weights, biases):
                fill = fill_bias

            svg += f'<circle cx="{cx}" cy="{cy}" r="{cr}" fill="{fill}" />'

            if has_bias and is_bias_neuron(i, j, weights, biases):
                svg += (
                    f'<text x="{cx}" y="{cy}" dominant-baseline="middle" '
                    f'text-anchor="middle" font-size="{cr*2/3}">+1</text>'
                )
            elif is_input(i, centers) or is_output(i, centers):
                var_name = 'x' if is_input(i, centers) else 'y'
                font_size = cr*2/3
                svg += (
                    f'<text x="{cx}" y="{cy}" dominant-baseline="middle" '
                    f'text-anchor="middle" font-size="{font_size}">{var_name}<tspan baseline-shift="sub" font-size="{font_size/2}">{j if len(layer_centers) > (1 + has_bias) else ""}</tspan></text>'
                )

    return svg


def draw_connections(centers, connect_stroke, weights, biases=None,
                     selected_weight=None, selected_stroke='lightblue'):
    svg = ''
    for i, layer_centers in enumerate(centers):
        for j, (cx, cy) in enumerate(layer_centers):
            i2 = i + 1
            if i2 >= len(centers):
                continue
            for j2, (cx2, cy2) in enumerate(centers[i2]):
                if (layer_has_bias_neuron(i2, biases)
                        and is_bias_neuron(i2, j2, weights, biases)):
                    continue

                if (layer_has_bias_neuron(i, biases)
                        and is_bias_neuron(i, j, weights, biases)):
                    if biases[i] is not None:
                        param = biases[i][j2]
                    else:
                        param = 0
                else:
                    param = weights[i][j2][j]
                connect_width = param_to_width(param)

                is_selected = (
                    selected_weight is not None
                    and i == selected_weight[0]
                    and j == selected_weight[1]
                    and j2 == selected_weight[2]
                )

                if is_selected:
                    stroke = selected_stroke
                else:
                    stroke = connect_stroke

                path_id = f'weight_{i}{j}{j2}'

                svg += (
                    f'<path d="M {cx} {cy} L {cx2} {cy2}" '
                    f'id="{path_id}" '
                    f'stroke="{stroke}" '
                    # f'stroke-dasharray="{10 if param < 0 else ""}" '
                    f'stroke-width="{connect_width}"/>'
                )

                if is_selected:
                    svg += (
                        '<text font-size="24">'
                        f'<textPath xlink:href="#{path_id}" startOffset="50%" '
                        f'text-anchor="middle">{param:.2f}</textPath>'
                        '</text>'
                    )

    return svg


def layer_has_bias_neuron(i, biases):
    return (
        biases is not None
        and len(biases) > i
        and biases[i] is not None
    )


def draw_mlp_from_params(
    weights,
    biases=None,
    selected_weight=None,
    cr=50,
    fill_input='green',
    fill_output='blue',
    fill_hidden='orange',
    fill_bias='lightgray',
    connect_stroke='black',
    h_pad=50,
    v_pad=25
):
    """
    Args:
        weights: List of weights per MLP layer. A weight matrix has shape M x L
            with M the number of neurons in the next layer and L the number of
            neurons in the previous layer.
        biases: List of biases per MLP layer. A bias vector has length M, with
            M the number of neurons in the next layer.
    """
    layers = [
        l.shape[1] + (
            1 if layer_has_bias_neuron(i, biases)
            else 0
        )
        for i, l in enumerate(weights)
    ]
    layers.append(weights[-1].shape[0])
    centers, svg_width, svg_height = get_neuron_centers(layers, cr, h_pad,
                                                        v_pad)
    svg = ''
    svg += draw_connections(centers, connect_stroke, weights, biases,
                            selected_weight)
    svg += draw_neurons(centers, cr, weights, biases, fill_input, fill_output,
                        fill_hidden, fill_bias)

    return svg, svg_width, svg_height

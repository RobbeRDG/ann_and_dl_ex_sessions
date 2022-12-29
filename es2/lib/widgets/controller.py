from functools import partial

import ipywidgets as widgets
from IPython.display import display

from ..models.mlp import MLP, ACTIV_FUNCS
from .mlp_response_view import MLPResponseView
from .mlp_arch_view import MLPArchView


class MLPController:
    def __init__(
        self,
        mlp=None,
        views=None,
        num_input_neurons=None,
        control_num_inputs=False,
        control_activation=True,
        control_hlayers=True,
        proj_3d=False
    ):
        if mlp is None:
            self._mlp = MLP(num_input_neurons=num_input_neurons)
        else:
            self._mlp = mlp

        if views is None:
            self.views = [
                MLPResponseView(self.mlp, proj_3d=proj_3d),
                MLPArchView(self.mlp)
            ]
        else:
            self.views = views

        self.control_num_inputs = control_num_inputs
        self.control_activation = control_activation
        self.control_hlayers = control_hlayers

        self.num_hidden_sliders = []

        self.widget = widgets.Output()
        self.debug_out = widgets.Output()
        self.info_out = widgets.Output()

        self.create_controller_widgets()

        self.render()

    @property
    def mlp(self):
        return self._mlp

    def set_data(self, data_x, data_y):
        if data_x is None and data_y is None:
            self.data_x = data_x
            self.data_y = data_x
            return

        if data_x.ndim == 1:
            data_x = data_x[..., None]
        if data_y.ndim == 1:
            data_y = data_y[..., None]

        if not data_x.shape[1] == self.mlp.num_input_neurons:
            raise ValueError(
                f"Input vectors should have length {self.mlp.num_input_neurons}, but have length {data_x.shape[1]}"
            )
        if not data_y.shape[1] == self.mlp.num_output_neurons:
            raise ValueError(
                f"Output vectors should have length {self.mlp.num_output_neurons}, but have length {data_y.shape[1]}"
            )
        self.data_x = data_x
        self.data_y = data_y

        for view in self.views:
            view.data_x = data_x
            view.data_y = data_y

        self.render_views()

    def render_views(self):
        with self.debug_out:
            for view in self.views:
                view.render()

    def get_controller_widget(self):
        return self.hparam_widgets

    def get_view_widget(self):
        return widgets.Box(
            [v.widget for v in self.views],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                align_items='center',
            )
        )

    def render(self):
        self.widget.clear_output()
        self.render_views()

        controller_region = self.get_controller_widget()
        view_region = self.get_view_widget()

        with self.widget:
            display(
                widgets.VBox([controller_region, view_region]),
                self.info_out,
                self.debug_out
            )

    def create_hparam_widgets(self):
        vbox_widgets = []

        if self.control_num_inputs:
            num_input_widget = widgets.BoundedIntText(
                value=self.mlp.num_input_neurons, min=1, max=2,
                description='Num input neurons'
            )
            num_input_widget.observe(
                partial(self.on_mlp_prop_change, 'num_input_neurons'),
                names='value'
            )
            vbox_widgets.append(num_input_widget)

        if self.control_activation:
            self.activ_widget = widgets.Dropdown(
                options=ACTIV_FUNCS,
                value=self.mlp.activation,
                description='Activation',
            )
            self.activ_widget.observe(
                partial(self.on_mlp_prop_change, 'activation'),
                names='value'
            )
            if len(self.num_hidden_sliders) > 0:
                vbox_widgets.append(self.activ_widget)

            output_activ_widget = widgets.Dropdown(
                options=ACTIV_FUNCS,
                value=self.mlp.output_activation,
                description='Output Activation',
            )
            output_activ_widget.observe(
                partial(self.on_mlp_prop_change, 'output_activation'),
                names='value'
            )
            vbox_widgets.append(output_activ_widget)

        if self.control_hlayers:
            add_hlayer_btn = widgets.Button(
                description='Add hidden layer'
            )
            add_hlayer_btn.on_click(self.on_add_hlayer)

            self.rm_hlayer_btn = widgets.Button(
                description='Remove hidden layer',
            )
            self.rm_hlayer_btn.on_click(self.on_rm_hlayer)

            vbox_widgets.extend([
                *self.num_hidden_sliders,
                add_hlayer_btn
            ])
            if len(self.num_hidden_sliders) > 0:
                vbox_widgets.append(self.rm_hlayer_btn)

        self.hparam_widgets = widgets.VBox(vbox_widgets)

    def create_controller_widgets(self):
        self.create_hparam_widgets()

    def on_num_hidden_neurs_change(self, hidden_layer, change):
        new_num_neurs = [*self.mlp.num_hidden_neurons]
        new_num_neurs[hidden_layer] = change['new']
        self.mlp.num_hidden_neurons = new_num_neurs

        self.create_controller_widgets()
        self.render()

    def on_add_hlayer(self, btn):
        with self.debug_out:
            num_hidden_slider = widgets.BoundedIntText(
                value=3, min=1, max=10,
                description=f'Num in HL {len(self.num_hidden_sliders) + 1}'
            )
            num_hidden_slider.observe(
                partial(self.on_num_hidden_neurs_change,
                        len(self.num_hidden_sliders)),
                names='value'
            )
            self.num_hidden_sliders.append(
                num_hidden_slider
            )
            self.rm_hlayer_btn.disabled = False
            self.activ_widget.disabled = False

            self.mlp.num_hidden_neurons = [
                *self.mlp.num_hidden_neurons,
                num_hidden_slider.value
            ]

            self.create_controller_widgets()
            self.render()

    def on_rm_hlayer(self, btn):
        del self.num_hidden_sliders[-1]
        self.mlp.num_hidden_neurons = self.mlp.num_hidden_neurons[:-1]

        if len(self.mlp.num_hidden_neurons) == 0:
            self.activ_widget.disabled = True
            self.rm_hlayer_btn.disabled = True

        self.create_controller_widgets()
        self.render()

    def on_mlp_prop_change(self, prop, change):
        with self.debug_out:
            setattr(self.mlp, prop, change['new'])
            self.create_controller_widgets()
            self.render()

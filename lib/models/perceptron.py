from .mlp import MLP


class Perceptron(MLP):
    def __init__(self, num_input_neurons):
        super().__init__(
            num_input_neurons=num_input_neurons,
            num_output_neurons=1,
            num_hidden_neurons=[],
            activation=None,
            output_activation='HardLimit',
        )


def create_perceptron():
    return Perceptron(2)

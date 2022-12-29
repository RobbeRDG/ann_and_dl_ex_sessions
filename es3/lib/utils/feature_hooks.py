from collections import OrderedDict

from torch import nn


def add_feature_hooks(model: nn.Module):
    """
    Return a dictionary that stores each layer's most recent output,
    using the layer's name as key.
    
    Args:
        model (nn.Module): The model to add the hooks to.
    """
    features = OrderedDict()

    def build_feature_hook(child_name):
        features.clear()
        def feature_hook(module, input, output):
            features[child_name] = output.detach().cpu()
        return feature_hook

    for name, module in model.named_children():
        module.register_forward_hook(build_feature_hook(name))

    return features
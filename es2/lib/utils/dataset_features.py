import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .feature_hooks import add_feature_hooks
from .model_device import get_model_device


def get_dataset_features(
    dataset: Dataset,
    model: nn.Module,
    layer_name: str,
    batch_size: int = 1,
    num_workers: int = 0
):
    """
    Return a batch with all outputs of a certain layer of a network
    for an entire dataset, along with the corresponding labels.

    Args:
        dataset (Dataset): The dataset. It is assumed that `__getitem__()` returns
            the model input and the corresponding label.
        model (nn.Module): The model.
        layer_name (str): The layer from which to collect the outputs.
        batch_size (int): The batch size with which we pass the dataset samples
            through the model.
        num_workers (int): The number of workers to use for the data loader.
    """
    model_features = add_feature_hooks(model)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    layer_features = []
    labels = []

    child_names = [
        name for name, child in model.named_children()
    ]

    if not layer_name in child_names:
        raise ValueError(
            f'Model has no layer named {layer_name}. '
            f'Available layers are: {", ".join(child_names)}'
        )

    device = get_model_device(model)

    for batch_x, batch_y in tqdm(data_loader):
        with torch.no_grad():
            model(batch_x.to(device))

        layer_features.append(
            model_features[layer_name].cpu()
        )
        labels.append(batch_y)

    return torch.cat(layer_features), torch.cat(labels)
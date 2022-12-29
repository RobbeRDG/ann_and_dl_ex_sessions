import torch


def get_accuracy(clf, x, y):
    y_pred = clf(x)
    return torch.div(
        (y_pred == y).sum(),
        len(y)
    )
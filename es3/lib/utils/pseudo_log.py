from itertools import product

import torch
from torch.utils.tensorboard import SummaryWriter

from .run_name import hparams_to_run_name


def create_pseudo_logs(log_root, num_steps = 100):
    """
    Generate logs of some fake training runs in the given directory.
    """
    for hparam1, hparam2 in product([0.01, 0.1, 1.0], [10, 20, 30]):
        hparams = {
            'hparam1': hparam1,
            'hparam2': hparam2
        }

        run_name = hparams_to_run_name(hparams)
        writer = SummaryWriter(log_dir=f'{log_root}/{run_name}')

        xs = torch.arange(num_steps).float()
        y1s = torch.exp(-xs*hparam1/hparam2) + torch.randn_like(xs)/100
        y2s = torch.sigmoid(xs*hparam1/hparam2) + torch.randn_like(xs)/100

        for x, y1, y2 in zip(xs, y1s, y2s):
            writer.add_scalar('metric1', y1, x)
            writer.add_scalar('metric2', y2, x)
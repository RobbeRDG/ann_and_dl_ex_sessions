from datetime import datetime
from logging import warning
from pathlib import Path
import re

from torch import nn


def get_optimizer_hparams(optimizer):
    def get_optimizer_hparam(hparam):
        if hparam in optimizer.defaults:
            return optimizer.defaults[hparam]

    return {
        'opt': optimizer.__class__.__name__,
        'lr': get_optimizer_hparam('lr'),
        'mom': get_optimizer_hparam('momentum'),
        'wd': get_optimizer_hparam('weight_decay'),
        'betas': get_optimizer_hparam('betas'),
    }


def get_dataloader_hparams(data_loader):
    def get_dl_hparam(hparam):
        if hasattr(data_loader, hparam):
            return getattr(data_loader, hparam)

    return {
        'bs': get_dl_hparam('batch_size')
    }


def hparams_to_str(hparams: dict):
    return "_".join(
        f'{name}({value})'
        for name, value in hparams.items()
        if value is not None
    )


def hparams_to_run_name(hparams: dict):
    timestamp = datetime.now().strftime('%Y-%M-%dT%H-%M-%S')
    return (
        f'{hparams_to_str(hparams)}_'
        f'{timestamp}'
    )


def get_optim_dl_hparams(optimizer, data_loader=None):
    hparams = get_optimizer_hparams(optimizer)

    if data_loader is not None:
        hparams.update(get_dataloader_hparams(data_loader))

    return hparams

def get_run_name(optimizer, data_loader=None):
    hparams = get_optim_dl_hparams(optimizer, data_loader=None)

    return hparams_to_run_name(hparams)


def get_ae_run_name(
    model,
    sparsity_weight, sparsity_rho,
    optimizer, data_loader=None
):
    run_name = (
        f'ae({model.num_input_neurons}-{model.num_hidden_neurons}-{model.num_input_neurons})_sw({sparsity_weight})_sr({sparsity_rho})_'
        + get_run_name(optimizer, data_loader)
    )
    return run_name


def get_num_in_features(model):
    if hasattr(model, 'children') and len(children := list(model.children())) > 0:
        return get_num_in_features(children[0])
    elif isinstance(model, nn.Linear):
        return model.in_features
    elif isinstance(model, nn.Conv2d):
        if isinstance(model.kernel_size, tuple):
            return (
                f'{model.in_channels}x{model.kernel_size[0]}x{model.kernel_size[1]}'
            )
        else:
            return (
                f'{model.in_channels}x{model.kernel_size}x{model.kernel_size}'
            )
    else:
        raise NotImplementedError


def get_num_out_features(model):
    if hasattr(model, 'children') and len(children := list(model.children())) > 0:
        if isinstance(children[-1], nn.Softmax):
            return get_num_out_features(children[-2])
        else:
            return get_num_out_features(children[-1])
    elif isinstance(model, nn.Linear):
        return model.out_features
    else:
        raise NotImplementedError


def get_clf_run_name(model, optimizer, data_loader=None):
    run_name = (
        f'clf({get_num_in_features(model)}-{get_num_out_features(model)})_'
        + get_run_name(optimizer, data_loader)
    )
    return run_name


def get_gan_hparams(netD, netG):
    return {
        'ndf': netD.num_features,
        'ngf': netG.num_features,
        'nz': netG.latent_size
    }


def get_gan_run_name(netD, netG, optimizer_D, optimizer_G, data_loader=None):
    hparams = {
        f'{k}D': v
        for k, v in get_optimizer_hparams(optimizer_D).items()
    }
    hparams.update({
        f'{k}G': v
        for k, v in get_optimizer_hparams(optimizer_G).items()
    })

    if data_loader is not None:
        hparams.update(get_dataloader_hparams(data_loader))

    hparams.update(get_gan_hparams(netD, netG))

    return f'gan_{hparams_to_run_name(hparams)}'


def run_name_to_dict(run_name):
    run_name = Path(run_name).name

    splits = run_name.split('_')
    hparams = {
        'run_name': run_name,
    }

    splits, timestamp = splits[:-1], splits[-1]
    time_ptrn = re.compile(r'\d+-\d+-\d{2}T\d{2}-\d{2}-\d{2}')
    if not time_ptrn.match(timestamp):
        warning(f'Expected the run name "{run_name}" to end with a timestamp, but didn\'t find one.')

    ptrn = re.compile(r'(?P<hparam>[^(]+)\((?P<value>[^)]+)\)')
    for split in splits:
        m = ptrn.match(split)
        if m is not None:
            value = m.group('value')
            try:
                value = int(value)
            except ValueError:
                pass
            try:
                value = float(value)
            except ValueError:
                pass
            hparams[m.group('hparam')] = value
        else:
            hparams[split] = True

    return hparams
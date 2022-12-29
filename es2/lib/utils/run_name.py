from datetime import datetime

from torch import nn


def get_run_name(optimizer, data_loader=None):
    timestamp = datetime.now().strftime('%Y-%M-%dT%H-%M-%S')

    def get_optimizer_hparam(hparam):
        if hparam in optimizer.defaults:
            return optimizer.defaults[hparam]

    def get_dl_hparam(hparam):
        if hasattr(data_loader, hparam):
            return getattr(data_loader, hparam)

    hparams = {
        'lr': get_optimizer_hparam('lr'),
        'mom': get_optimizer_hparam('momentum'),
        'wd': get_optimizer_hparam('weight_decay'),
    }

    if data_loader is not None:
        hparams['bs'] = get_dl_hparam('batch_size')

    hparam_str = "_".join(
        f'{name}({value})'
        for name, value in hparams.items()
        if value is not None
    )

    return (
        f'opt({optimizer.__class__.__name__})'
        f'_{hparam_str}'
        f'_{timestamp}'
    )


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
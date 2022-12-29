import torch


def create_time_series_inputs_targets(time_series: torch.Tensor, lag: int):
    """
    Creates inputs (each of length `lag`) and the respective targets from a time series.

    We shift a sliding window of size `lag` and stride 1 over the time
    series. The elements covered by the window are appended to `inputs`,
    the first element after the window is appended to `targets`.

    Args:
        time_series (torch.Tensor): The time series (1D tensor).
        lag (int): The lag
    Return:
        The inputs and targets.
    """
    if lag <= 0:
        raise ValueError(f'Lag should be > 0, got {lag}')

    num_samples = len(time_series) - lag

    start_idxs = torch.arange(num_samples)
    input_idxs = start_idxs[None, :].T + torch.arange(lag)
    inputs = time_series[input_idxs]

    target_idxs = input_idxs[:, -1] + 1
    targets = time_series[target_idxs][:, None]

    return inputs, targets
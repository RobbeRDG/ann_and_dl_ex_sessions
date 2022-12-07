from pathlib import Path
import torch


def load_santa_fe():
    """
    Return Santa Fe training and test set.
    """
    dataset = []
    path = Path(__file__).parent

    for img in ['lasertrain.dat', 'laserpred.dat']:
        dataset.append(torch.tensor([
            eval(x) for x in (path / img).read_text().splitlines()
        ]))
    return dataset
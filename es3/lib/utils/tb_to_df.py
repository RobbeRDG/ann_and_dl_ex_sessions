from collections import defaultdict
from pathlib import Path
import time

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from .run_name import run_name_to_dict


def tb_to_df(log_root: str):
    """
    Convert Tensorboard logs to pandas DataFrames.
    """
    log_root = Path(log_root)

    runs = [EventAccumulator(str(log_dir)).Reload()
            for log_dir in log_root.glob('*')
            if not log_dir.name.startswith(".")]

    results = []

    for run in runs:
        hparams = run_name_to_dict(run.path)

        steps = defaultdict(dict)

        for key in run.scalars.Keys():
            for e in run.Scalars(key):
                steps[e.step].update({
                    key: e.value,
                    f'wall_time ({key})': e.wall_time,
                })

        for step, values in steps.items():
            results.append({
                'step': step,
                **values,
                **hparams
            })

    df = pd.DataFrame(results)
    time_cols = [col for col in df if col.startswith('wall_time')]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], unit='s')
    return df
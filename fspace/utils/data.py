import os
import logging
from pathlib import Path
import torch
from torch.utils.data import random_split


def get_data_dir(data_dir=None):
    if data_dir is None:
        if os.environ.get('DATADIR') is not None:
            data_dir = os.environ.get('DATADIR')
            logging.debug(f'Using default data directory from environment "{data_dir}".')
        else:
            home_data_dir = Path().home() / 'datasets'
            data_dir = str(home_data_dir.resolve())
            logging.debug(f'Using default HOME data directory "{data_dir}".')

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    return data_dir


def train_test_split(dataset, test_size=.2, seed=None):
    N = len(dataset)
    N_test = int(test_size * N)
    N -= N_test

    if seed is not None:
        train, test = random_split(dataset, [N, N_test], 
                                   generator=torch.Generator().manual_seed(seed))
    else:
        train, test = random_split(dataset, [N, N_test])

    return train, test

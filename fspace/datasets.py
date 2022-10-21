import logging
import numpy as np
import torch
from torch.utils.data import Subset
from timm.data import create_dataset
import torchvision.transforms as transforms

from .utils.data import get_data_dir, train_test_split


def get_fmnist(root=None, seed=42, **_):
    _FMNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,)),
        transforms.Lambda(lambda img: img.permute(1, 2, 0)) ## Convert to HxWxC from CxHxW
    ])

    train_data = create_dataset('torch/fashion_mnist', root=root, split='train',
                                transform=_FMNIST_TRANSFORM, download=True)

    train_data, val_data = train_test_split(train_data, test_size=.1, seed=seed)

    test_data = create_dataset('torch/fashion_mnist', root=root, split='test',
                               transform=_FMNIST_TRANSFORM, download=True)

    return train_data, val_data, test_data


_DATASET_CFG = {
    'fmnist': {
        'num_classes': 10,
        'get_fn': get_fmnist,
    },
}


def get_dataset(dataset, root=None, train_subset=1, **kwargs):
    assert dataset in _DATASET_CFG, f'Dataset "{dataset}" not supported'

    root = get_data_dir(data_dir=root)

    train_data, val_data, test_data = _DATASET_CFG[dataset].get('get_fn')(root=root, **kwargs)

    num_classes = _DATASET_CFG[dataset].get('num_classes')

    if np.abs(train_subset) < 1:
        n = len(train_data)
        ns = int(n * np.abs(train_subset))

        ## NOTE: -ve train_subset fraction to get latter segment.
        randperm = torch.randperm(n)
        randperm = randperm[ns:] if train_subset < 0 else randperm[:ns]

        train_data = Subset(train_data, randperm)

    setattr(train_data, 'n_classes', num_classes)

    logging.info(f'Train Dataset Size: {len(train_data)};  Test Dataset Size: {len(test_data)}')

    return train_data, val_data, test_data

import logging
import numpy as np
import torch
from torch.utils.data import Subset
from timm.data import create_dataset
import torchvision.transforms as transforms

from .utils.data import get_data_dir, train_test_split, LabelNoiseDataset


## Convert from CxHxW to HxWxC for Flax.
chw2hwc_fn = lambda img: img.permute(1, 2, 0)


def get_fmnist(root=None, seed=42, val_size=1/6, **_):
    _FMNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2861,), (0.3530,)),
        transforms.Lambda(chw2hwc_fn)
    ])

    train_data = create_dataset('torch/fashion_mnist', root=root, split='train',
                                transform=_FMNIST_TRANSFORM, download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    test_data = create_dataset('torch/fashion_mnist', root=root, split='test',
                               transform=_FMNIST_TRANSFORM, download=True)

    return train_data, val_data, test_data


def get_cifar10(root=None, seed=42, val_size=.1, **_):
    _CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4465), (.247, .243, .261)),
        transforms.Lambda(chw2hwc_fn)
    ])
    _CIFAR10_TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4465), (.247, .243, .261)),
        transforms.Lambda(chw2hwc_fn)
    ])

    train_data = create_dataset('torch/cifar10', root=root, split='train',
                                transform=_CIFAR10_TRAIN_TRANSFORM, download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    test_data = create_dataset('torch/cifar10', root=root, split='test',
                               transform=_CIFAR10_TEST_TRANSFORM, download=True)

    return train_data, val_data, test_data


_DATASET_CFG = {
    'fmnist': {
        'n_classes': 10,
        'get_fn': get_fmnist,
    },
    'cifar10': {
        'n_classes': 10,
        'get_fn': get_cifar10,
    },
}


def get_dataset(dataset, root=None, seed=42, train_subset=1, label_noise=0, **kwargs):
    assert dataset in _DATASET_CFG, f'Dataset "{dataset}" not supported'

    root = get_data_dir(data_dir=root)

    train_data, val_data, test_data = _DATASET_CFG[dataset].get('get_fn')(root=root, seed=seed, **kwargs)

    n_classes = _DATASET_CFG[dataset].get('n_classes')

    if label_noise > 0:
        train_data = LabelNoiseDataset(
            train_data, n_labels=n_classes, label_noise=label_noise, seed=seed)

    if np.abs(train_subset) < 1:
        n = len(train_data)
        ns = int(n * np.abs(train_subset))

        ## NOTE: -ve train_subset fraction to get latter segment.
        randperm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        randperm = randperm[ns:] if train_subset < 0 else randperm[:ns]

        train_data = Subset(train_data, randperm)

    setattr(train_data, 'n_classes', n_classes)

    logging.info(f'Train Dataset Size: {len(train_data)};  Test Dataset Size: {len(test_data)}')

    return train_data, val_data, test_data

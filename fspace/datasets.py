from functools import partial
from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import Subset
from timm.data import create_dataset
import torchvision.transforms as transforms

from .utils.data import get_data_dir, train_test_split, LabelNoiseDataset


## Convert from CxHxW to HxWxC for Flax.
chw2hwc_fn = lambda img: img.permute(1, 2, 0)


def get_mnist(root=None, seed=42, val_size=1/6, normalize=None, **_):
    _MNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.Lambda(chw2hwc_fn)
    ])

    train_data = create_dataset('torch/mnist', root=root, split='train',
                                transform=_MNIST_TRANSFORM, download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    test_data = create_dataset('torch/mnist', root=root, split='test',
                               transform=_MNIST_TRANSFORM, download=True)

    return train_data, val_data, test_data


def get_fmnist(root=None, seed=42, val_size=1/6, normalize=None, **_):
    _FMNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
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


def get_kmnist(root=None, seed=42, val_size=1/6, normalize=None, **_):
    _FMNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.Lambda(chw2hwc_fn)
    ])

    train_data = create_dataset('torch/kmnist', root=root, split='train',
                                transform=_FMNIST_TRANSFORM, download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    test_data = create_dataset('torch/kmnist', root=root, split='test',
                               transform=_FMNIST_TRANSFORM, download=True)

    return train_data, val_data, test_data



def get_cifar10(root=None, seed=42, val_size=0., normalize=None,
                v1=False, corrupted=False, batch_size=128, **_):
    _CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.Lambda(chw2hwc_fn)
    ])
    _CIFAR10_TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.Lambda(chw2hwc_fn)
    ])

    train_data = create_dataset('torch/cifar10', root=root, split='train',
                                transform=_CIFAR10_TRAIN_TRANSFORM, download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    if v1:
        test_data = create_dataset('tfds/cifar10_1', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=_CIFAR10_TEST_TRANSFORM, download=True)
    elif corrupted:
        test_data = create_dataset('tfds/cifar10_corrupted', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=_CIFAR10_TEST_TRANSFORM, download=True)
    else:
        test_data = create_dataset('torch/cifar10', root=root, split='test',
                                    transform=_CIFAR10_TEST_TRANSFORM, download=True)

    return train_data, val_data, test_data


def get_svhn(root=None, seed=42, val_size=0., normalize=None, **_):
    '''Dataset SVHN

    root (str): Root directory where 'svhn' folder exists or will be downloaded to.
    '''

    from torchvision.datasets import SVHN

    _SVHN_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.Lambda(chw2hwc_fn)
    ])

    (Path(root) / 'svhn').mkdir(parents=True, exist_ok=True)

    train_data = SVHN(root=Path(root) / 'svhn', split='train',
                      transform=_SVHN_TRANSFORM, download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    test_data = SVHN(root=Path(root) / 'svhn', split='test',
                     transform=_SVHN_TRANSFORM, download=True)

    return train_data, val_data, test_data


def get_eyepacs(root=None, batch_size=128, **_):
    '''

    See https://github.com/google/uncertainty-baselines/tree/main/baselines/diabetic_retinopathy_detection#data-installation.
    
    `root` assumes an "EyePACS" folder containing the manual download as instructed above.
    '''
    _EYEPACS_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(chw2hwc_fn)
    ])
    
    root = Path(root) / 'EyePACS'

    train_data = create_dataset('tfds/diabetic_retinopathy_detection', root=root, split='train',
                                is_training=True, batch_size=batch_size, 
                                transform=_EYEPACS_TRANSFORM, download=True)

    test_data = create_dataset('tfds/diabetic_retinopathy_detection', root=root, split='test',
                                is_training=True, batch_size=batch_size,
                                transform=_EYEPACS_TRANSFORM, download=True)

    return train_data, None, test_data


_DATASET_CFG = {
    'mnist': {
        'num_classes': 10,
        'get_fn': get_mnist,
        'normalize': [(0.1307,), (0.3081,)],
    },
    'svhn': {
        'n_classes': 10,
        'get_fn': get_svhn,
        'normalize': [(.5, .5, .5), (.25, .25, .25)],
    },
    'fmnist': {
        'n_classes': 10,
        'get_fn': get_fmnist,
        'normalize': [(0.2861,), (0.3530,)],
    },
    'kmnist': {
        'n_classes': 10,
        'get_fn': get_kmnist,
        'normalize': [(0.5,), (0.5,)],
    },
    'cifar10': {
        'n_classes': 10,
        'get_fn': get_cifar10,
        'normalize': [(.4914, .4822, .4465), (.247, .243, .261)],
    },
    'cifar10_1': {
        'n_classes': 10,
        'get_fn': partial(get_cifar10, v1=True),
        'ctx_idx': -1,
        'normalize': [(.4914, .4822, .4465), (.247, .243, .261)],
    },
    'cifar10c': {
        'n_classes': 10,
        'get_fn': partial(get_cifar10, corrupted=True),
        'ctx_idx': -1,
        'normalize': [(.4914, .4822, .4465), (.247, .243, .261)],
    },
    'eyepacs': {
        'n_classes': 5,
        'get_fn': get_eyepacs,
    },
}


def get_dataset_normalization(dataset):
    assert dataset in _DATASET_CFG, f'Dataset "{dataset}" not supported'

    return _DATASET_CFG[dataset].get('normalize')


def get_dataset(dataset, root=None, seed=42, train_subset=1, label_noise=0, is_ctx=False, **kwargs):
    assert dataset in _DATASET_CFG, f'Dataset "{dataset}" not supported'

    root = get_data_dir(data_dir=root)

    all_kwargs = { **_DATASET_CFG[dataset], **kwargs }
    raw_data = _DATASET_CFG[dataset].get('get_fn')(root=root, seed=seed, **all_kwargs)

    ## 
    # If dataset used for context points, 
    # then only return the desired split through tuple's index.
    # Default: 0 (i.e. train split)
    #
    if is_ctx:
        return raw_data[_DATASET_CFG[dataset].get('ctx_idx', 0)]

    train_data, val_data, test_data = raw_data

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
    if val_data is not None:
        setattr(val_data, 'n_classes', n_classes)
    if test_data is not None:
        setattr(test_data, 'n_classes', n_classes)

    logging.info(f'Train Dataset Size: {len(train_data)};  Test Dataset Size: {len(test_data)}')

    return train_data, val_data, test_data

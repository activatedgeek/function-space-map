import logging
from functools import partial
import numpy as np
import torch
from torch.utils.data import Subset


from .utils import get_data_dir, LabelNoiseDataset
from .factory import \
    get_mnist, \
    get_svhn, \
    get_fmnist, \
    get_kmnist, \
    get_cifar10, \
    get_cifar100
from .two_moons import get_twomoons
from .third_party.aptos import get_aptos_orig
from .third_party.cassava import get_cassava_orig
from .third_party.melanoma import get_melanoma_orig


_DATASET_CFG = {
    'twomoons': {
        'n_classes': 2,
        'get_fn': get_twomoons,
        'random_state': 137,
        'noise': 5e-2,
        'normalize': [[[0.49983146, 0.24929603]], [[0.87234487, 0.48621055]]], ## Update stats if random state and noise updated.
    },
    'mnist': {
        'n_classes': 10,
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
        'normalize': [(.4914, .4822, .4465), (.2023, .1994, .2010)],
    },
    'cifar10_1': {
        'n_classes': 10,
        'get_fn': partial(get_cifar10, v1=True),
        'normalize': [(.4914, .4822, .4465), (.2023, .1994, .2010)],
    },
    'cifar100': {
        'n_classes': 100,
        'get_fn': get_cifar100,
        'normalize': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
    },
    'aptos': {
        'n_classes': 5,
        'get_fn': get_aptos_orig,
        'normalize': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    },
    'cassava': {
        'n_classes': 5,
        'get_fn': get_cassava_orig,
        'normalize': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    },
    'melanoma': {
        'n_classes': 2,
        'get_fn': get_melanoma_orig,
        'normalize': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    },
}


def get_dataset_normalization(dataset):
    assert dataset in _DATASET_CFG, f'Dataset "{dataset}" not supported'

    return _DATASET_CFG[dataset].get('normalize')


def get_dataset(dataset, root=None, seed=42, train_subset=1, label_noise=0, is_ctx=False, **kwargs):
    assert dataset in _DATASET_CFG, f'Dataset "{dataset}" not supported'

    root = get_data_dir(data_dir=root)

    all_kwargs = { 'augment': not is_ctx, **_DATASET_CFG[dataset], **kwargs }
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
    setattr(test_data, 'n_classes', n_classes)

    logging.info(f'Train Dataset Size: {len(train_data)};  Test Dataset Size: {len(test_data)}')

    return train_data, val_data, test_data

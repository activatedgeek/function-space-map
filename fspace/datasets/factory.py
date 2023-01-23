from pathlib import Path
from timm.data import create_dataset
import torchvision.transforms as transforms

from .utils import train_test_split


## Convert from CxHxW to HxWxC for Flax.
chw2hwc_fn = lambda img: img.permute(1, 2, 0)

_CIFAR_AUGMENT_TRANSFORM = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
]


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



def get_cifar10(root=None, seed=42, val_size=0., normalize=None, augment=True,
                v1=False, corrupted=False, batch_size=128, **_):
    _TEST_TRANSFORM = [
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.Lambda(chw2hwc_fn)
    ]

    _TRAIN_TRANSFORM = (_CIFAR_AUGMENT_TRANSFORM if augment else []) + _TEST_TRANSFORM

    train_data = create_dataset('torch/cifar10', root=root, split='train',
                                transform=transforms.Compose(_TRAIN_TRANSFORM), download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    if v1:
        test_data = create_dataset('tfds/cifar10_1', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    elif corrupted:
        test_data = create_dataset('tfds/cifar10_corrupted', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    else:
        test_data = create_dataset('torch/cifar10', root=root, split='test',
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)

    return train_data, val_data, test_data


def get_cifar100(root=None, seed=42, val_size=0., normalize=None, augment=True, **_):
    _TEST_TRANSFORM = [
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        transforms.Lambda(chw2hwc_fn)
    ]

    _TRAIN_TRANSFORM = (_CIFAR_AUGMENT_TRANSFORM if augment else []) + _TEST_TRANSFORM

    train_data = create_dataset('torch/cifar100', root=root, split='train',
                                transform=transforms.Compose(_TRAIN_TRANSFORM), download=True)

    if val_size > 0.:
        train_data, val_data = train_test_split(train_data, test_size=val_size, seed=seed)
    else:
        val_data = None

    test_data = create_dataset('torch/cifar100', root=root, split='test',
                                transform=transforms.Compose(_TEST_TRANSFORM), download=True)

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

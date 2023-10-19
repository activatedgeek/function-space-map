from timm.data import create_dataset
import torchvision.transforms as transforms

from .registry import register_dataset
from .utils import train_test_split

__all__ = [
    "get_mnist",
    "get_fmnist",
    "get_kmnist",
]


__MNIST_ATTRS = dict(num_classes=10, normalize=((0.1307,), (0.3081,)))


def get_mnist(
    root=None, seed=42, val_size=1 / 6, normalize=None, channels_last=False, **_
):
    normalize = normalize or __MNIST_ATTRS.get("normalize")

    _MNIST_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*normalize),
            *(
                [transforms.Lambda(lambda x: x.permute(1, 2, 0))]
                if channels_last
                else []
            ),
        ]
    )

    train_data = create_dataset(
        "torch/mnist",
        root=root,
        split="train",
        transform=_MNIST_TRANSFORM,
        download=True,
    )

    if val_size > 0.0:
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, seed=seed
        )
    else:
        val_data = None

    test_data = create_dataset(
        "torch/mnist",
        root=root,
        split="test",
        transform=_MNIST_TRANSFORM,
        download=True,
    )

    return train_data, val_data, test_data


__FMNIST_ATTRS = dict(num_classes=10, normalize=((0.2861,), (0.3530,)))


def get_fmnist(
    root=None, seed=42, val_size=1 / 6, normalize=None, channels_last=False, **_
):
    normalize = normalize or __FMNIST_ATTRS.get("normalize")

    _FMNIST_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*normalize),
            *(
                [transforms.Lambda(lambda x: x.permute(1, 2, 0))]
                if channels_last
                else []
            ),
        ]
    )

    train_data = create_dataset(
        "torch/fashion_mnist",
        root=root,
        split="train",
        transform=_FMNIST_TRANSFORM,
        download=True,
    )

    if val_size > 0.0:
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, seed=seed
        )
    else:
        val_data = None

    test_data = create_dataset(
        "torch/fashion_mnist",
        root=root,
        split="test",
        transform=_FMNIST_TRANSFORM,
        download=True,
    )

    return train_data, val_data, test_data


__KMNIST_ATTRS = dict(num_classes=10, normalize=((0.5,), (0.5,)))


def get_kmnist(
    root=None, seed=42, val_size=1 / 6, normalize=None, channels_last=False, **_
):
    normalize = normalize or __KMNIST_ATTRS.get("normalize")

    _FMNIST_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*normalize),
            *(
                [transforms.Lambda(lambda x: x.permute(1, 2, 0))]
                if channels_last
                else []
            ),
        ]
    )

    train_data = create_dataset(
        "torch/kmnist",
        root=root,
        split="train",
        transform=_FMNIST_TRANSFORM,
        download=True,
    )

    if val_size > 0.0:
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, seed=seed
        )
    else:
        val_data = None

    test_data = create_dataset(
        "torch/kmnist",
        root=root,
        split="test",
        transform=_FMNIST_TRANSFORM,
        download=True,
    )

    return train_data, val_data, test_data


@register_dataset(attrs=__MNIST_ATTRS)
def mnist(*args, **kwargs):
    return get_mnist(*args, **kwargs)


@register_dataset(attrs=__FMNIST_ATTRS)
def fmnist(*args, **kwargs):
    return get_fmnist(*args, **kwargs)


@register_dataset(attrs=__KMNIST_ATTRS)
def kmnist(*args, **kwargs):
    return get_kmnist(*args, **kwargs)

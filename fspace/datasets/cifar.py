from timm.data import create_dataset
import torchvision.transforms as transforms

from .registry import register_dataset
from .utils import train_test_split


__all__ = [
    "get_cifar10",
    "get_cifar100",
]


_CIFAR_AUGMENT_TRANSFORM = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
]

__CIFAR10_ATTRS = dict(
    num_classes=10, normalize=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
)


def get_cifar10(
    root=None,
    seed=42,
    val_size=0.0,
    normalize=None,
    augment=True,
    channels_last=False,
    v1=False,
    corr_config=None,
    batch_size=128,
    resize=32,
    **_,
):
    normalize = normalize or __CIFAR10_ATTRS.get("normalize")

    _TEST_TRANSFORM = [
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        *([transforms.Lambda(lambda x: x.permute(1, 2, 0))] if channels_last else []),
    ]

    _TRAIN_TRANSFORM = (_CIFAR_AUGMENT_TRANSFORM if augment else []) + _TEST_TRANSFORM

    train_data = create_dataset(
        "torch/cifar10",
        root=root,
        split="train",
        transform=transforms.Compose(_TRAIN_TRANSFORM),
        download=True,
    )

    if val_size > 0.0:
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, seed=seed
        )
    else:
        val_data = None

    if v1:
        test_data = create_dataset(
            "tfds/cifar10_1",
            root=root,
            split="test",
            is_training=True,
            batch_size=batch_size,
            transform=transforms.Compose(_TEST_TRANSFORM),
            download=True,
        )
    elif corr_config is not None:
        ##
        # NOTE: Modify timm/data/parsers/parser_factory.py#L9 manually.
        # See https://github.com/huggingface/pytorch-image-models/issues/1598#issuecomment-1362207883.
        #
        test_data = create_dataset(
            f"tfds/cifar10_corrupted/{corr_config}",
            root=root,
            split="test",
            is_training=True,
            batch_size=batch_size,
            transform=transforms.Compose(_TEST_TRANSFORM),
            download=True,
        )
    else:
        test_data = create_dataset(
            "torch/cifar10",
            root=root,
            split="test",
            transform=transforms.Compose(_TEST_TRANSFORM),
            download=True,
        )

    return train_data, val_data, test_data


__CIFAR100_ATTRS = dict(
    num_classes=100, normalize=((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
)


def get_cifar100(
    root=None,
    seed=42,
    val_size=0.0,
    normalize=None,
    augment=True,
    channels_last=False,
    resize=32,
    **_,
):
    normalize = normalize or __CIFAR100_ATTRS.get("normalize")

    _TEST_TRANSFORM = [
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
        *([transforms.Lambda(lambda x: x.permute(1, 2, 0))] if channels_last else []),
    ]

    _TRAIN_TRANSFORM = (_CIFAR_AUGMENT_TRANSFORM if augment else []) + _TEST_TRANSFORM

    train_data = create_dataset(
        "torch/cifar100",
        root=root,
        split="train",
        transform=transforms.Compose(_TRAIN_TRANSFORM),
        download=True,
    )

    if val_size > 0.0:
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, seed=seed
        )
    else:
        val_data = None

    test_data = create_dataset(
        "torch/cifar100",
        root=root,
        split="test",
        transform=transforms.Compose(_TEST_TRANSFORM),
        download=True,
    )

    return train_data, val_data, test_data


@register_dataset(attrs=__CIFAR10_ATTRS)
def cifar10(*args, **kwargs):
    return get_cifar10(*args, **kwargs)


@register_dataset(attrs=__CIFAR10_ATTRS)
def cifar10_224(*args, **kwargs):
    kwargs["normalize"] = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    return get_cifar10(*args, **kwargs, resize=224)


@register_dataset(attrs=__CIFAR10_ATTRS)
def cifar10_1(*args, **kwargs):
    all_kwargs = {**kwargs, "v1": True}
    return get_cifar10(*args, **all_kwargs)


@register_dataset(attrs=__CIFAR10_ATTRS)
def cifar10c(*args, corr_config=None, **kwargs):
    assert corr_config is not None, "Missing corruption configuration name."
    all_kwargs = {**kwargs, "corr_config": corr_config}
    return get_cifar10(*args, **all_kwargs)


@register_dataset(attrs=__CIFAR100_ATTRS)
def cifar100(*args, **kwargs):
    return get_cifar100(*args, **kwargs)


@register_dataset(attrs=__CIFAR100_ATTRS)
def cifar100_224(*args, **kwargs):
    kwargs["normalize"] = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    return get_cifar100(*args, **kwargs, resize=224)

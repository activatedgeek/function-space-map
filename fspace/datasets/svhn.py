from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import SVHN

from .registry import register_dataset
from .utils import train_test_split


__all__ = [
    "get_svhn",
]


__SVHN_ATTRS = dict(num_classes=10, normalize=((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)))


def get_svhn(
    root=None, seed=42, val_size=0.0, normalize=None, channels_last=False, **_
):
    """Dataset SVHN

    root (str): Root directory where 'svhn' folder exists or will be downloaded to.
    """
    normalize = normalize or __SVHN_ATTRS.get("normalize")

    _SVHN_TRANSFORM = transforms.Compose(
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

    (Path(root) / "svhn").mkdir(parents=True, exist_ok=True)

    train_data = SVHN(
        root=Path(root) / "svhn",
        split="train",
        transform=_SVHN_TRANSFORM,
        download=True,
    )

    if val_size > 0.0:
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, seed=seed
        )
    else:
        val_data = None

    test_data = SVHN(
        root=Path(root) / "svhn", split="test", transform=_SVHN_TRANSFORM, download=True
    )

    return train_data, val_data, test_data


@register_dataset(attrs=__SVHN_ATTRS)
def svhn(*args, **kwargs):
    return get_svhn(*args, **kwargs)

from .registry import register_dataset, get_dataset, get_dataset_attrs, list_datasets
from .utils import IndexedDataset, LabelNoiseDataset, get_loader

__all__ = [
    "register_dataset",
    "get_dataset",
    "get_dataset_attrs",
    "list_datasets",
    "IndexedDataset",
    "LabelNoiseDataset",
    "get_loader",
]


def __setup():
    from importlib import import_module

    for n in [
        "cifar",
        "mnist",
        "svhn",
        "twomoons",
        "whitenoise",
    ]:
        import_module(f".{n}", __name__)


__setup()

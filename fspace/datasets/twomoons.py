import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
from torchvision import transforms


from .registry import register_dataset

__all__ = [
    "get_twomoons",
]


class TwoMoonsDataset(Dataset):
    def __init__(
        self, n_samples, split="train", noise=5e-2, random_state=137, transform=None
    ):
        super().__init__()

        if split == "train":
            self.X, self.y = make_moons(
                n_samples=n_samples, noise=noise, random_state=random_state
            )
        elif split == "test":
            x, y = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
            xx, yy = np.meshgrid(x, y)
            self.X = np.array((xx.ravel(), yy.ravel())).T
            self.y = np.zeros(len(self.X))  ## Dummy

        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform is not None:
            x = self.transform(x[None, ...])[0, 0]
        return x, self.y[index]

    def __len__(self):
        return len(self.X)


__TWOMOONS_ATTRS = dict(num_classes=2, normalize=((0.0,), (1.0,)))


def get_twomoons(n_samples=100, normalize=None, noise=None, random_state=None, **_):
    normalize = normalize or __TWOMOONS_ATTRS.get("normalize")

    mean, std = normalize
    mean, std = torch.Tensor(mean), torch.Tensor(std)
    _TEST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: (x - mean) / (std + 1e-6))]
    )

    train_data = TwoMoonsDataset(
        n_samples,
        split="train",
        noise=noise,
        random_state=random_state,
        transform=_TEST_TRANSFORM,
    )
    test_data = TwoMoonsDataset(n_samples, split="test", transform=_TEST_TRANSFORM)

    return train_data, None, test_data


@register_dataset(attrs=__TWOMOONS_ATTRS, random_state=0, noise=0.2)
def twomoons(*args, **kwargs):
    return get_twomoons(*args, **kwargs)

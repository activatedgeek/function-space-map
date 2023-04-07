import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
from torchvision import transforms


class TwoMoonsDataset(Dataset):
    def __init__(self, n_samples, split='train', noise=5e-2, random_state=137, transform=None):
        super().__init__()

        if split == 'train':
            self.X, self.y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        elif split == 'test' or split == 'context':
            x, y = np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)
            xx, yy = np.meshgrid(x, y)
            self.X = np.array((xx.ravel(), yy.ravel())).T
            self.y = np.zeros(len(self.X))  ## Dummy

        if split == 'context':
            idx = np.random.choice(len(self.X), size=n_samples, replace=False)
            self.X = self.X[idx]
            self.y = self.y[idx]

        self.transform = transform
    
    def __getitem__(self, index):
        x = self.X[index]
        if self.transform is not None:
            x = self.transform(x[None, ...])[0,0]
        return x, self.y[index]
    
    def __len__(self):
        return len(self.X)


def get_twomoons(n_samples=200, normalize=None, noise=None, seed=None, **_):
    mean, std = normalize
    mean, std = torch.Tensor(mean), torch.Tensor(std)
    _TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - mean) / (std + 1e-6))
    ])

    train_data = TwoMoonsDataset(n_samples, split='train', noise=noise, random_state=seed,
                                 transform=_TEST_TRANSFORM)
    test_data = TwoMoonsDataset(n_samples, split='test',
                                transform=_TEST_TRANSFORM)

    return train_data, None, test_data


def get_twomoons_ctx(n_samples=10, normalize=None, random_state=None, **_):
    mean, std = normalize
    mean, std = torch.Tensor(mean), torch.Tensor(std)
    _TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - mean) / (std + 1e-6))
    ])

    ctx_data = TwoMoonsDataset(n_samples, split='context', random_state=random_state,
                                 transform=_TEST_TRANSFORM)

    return ctx_data

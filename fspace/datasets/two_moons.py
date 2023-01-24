import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons


class TwoMoonsDataset(Dataset):
    def __init__(self, n_samples, seed=42, noise=5e-2, split='train'):
        super().__init__()

        if split == 'train':
            self.X, self.y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
            
            self.X = torch.from_numpy(self.X)
            self.y = torch.from_numpy(self.y)
        elif split == 'test':
            X, y = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
            xx, yy = np.meshgrid(X, y)
            self.X = torch.from_numpy(np.array((xx.ravel(), yy.ravel())).T)
            self.y = torch.zeros(len(self.X))
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)


def get_twomoons(n_samples=100, seed=42, **_):
    '''
    @NOTE: train and test data are the same.
    '''
    train_data = TwoMoonsDataset(n_samples, seed=seed, split='train')
    test_data = TwoMoonsDataset(n_samples, seed=seed, split='test')

    return train_data, None, test_data

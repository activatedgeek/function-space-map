import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons


class TwoMoonsDataset(Dataset):
    def __init__(self, n_samples, noise=.1, seed=42):
        super().__init__()

        self.X, self.y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)


def get_twomoons(n_samples=100, noise=.1, **_):
    '''
    @NOTE: train and test data are the same.
    '''
    train_data = TwoMoonsDataset(n_samples, noise=noise)

    return train_data, None, train_data

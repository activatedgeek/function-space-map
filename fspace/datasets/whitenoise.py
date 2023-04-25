import torch
from torch.utils.data import Dataset


class WhiteNoiseDataset(Dataset):
    def __init__(self, ref_tensor, n=100, seed=None):
        super().__init__()

        self.X = torch.randn(n, *ref_tensor.shape,
                             generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        ## Return 0 for backward compatibility.
        return self.X[index], 0


def get_white_noise(ref_tensor=None, num_samples=None, seed=None, **_):
    ## No val/test splits.
    train_data = WhiteNoiseDataset(ref_tensor, n=num_samples, seed=seed)
    return train_data, None, train_data

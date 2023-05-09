import torch
from torch.utils.data import Dataset

from .registry import register_dataset

__all__ = [
    'get_whitenoise',
]


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


__NOISE_ATTRS = dict(num_classes=None)

def get_whitenoise(ref_tensor=None, num_samples=None, seed=None, **_):
    ## No val/test splits.
    train_data = WhiteNoiseDataset(ref_tensor, n=num_samples, seed=seed)
    return train_data, None, train_data


@register_dataset(attrs=__NOISE_ATTRS, seed=2651, num_samples=10000)
def whitenoise(*args, **kwargs):
    return get_whitenoise(*args, **kwargs)

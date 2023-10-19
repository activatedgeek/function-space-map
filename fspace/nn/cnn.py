from flax import linen as nn

from .registry import register_model


class SmallCNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


class TinyCNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Conv(features=8, kernel_size=(3, 3))(x)
        x = nn.elu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=8, kernel_size=(3, 3))(x)
        x = nn.elu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=32)(x)
        x = nn.elu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


@register_model
def smallcnn(**kwargs):
    return SmallCNN(**kwargs)


@register_model
def tinycnn(**kwargs):
    return TinyCNN(**kwargs)

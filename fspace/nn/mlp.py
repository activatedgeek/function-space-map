from functools import partial
from flax import linen as nn
from typing import Callable

from .registry import register_model


class TinyMLP(nn.Module):
  num_classes: int
  hidden_size: int
  n_layers: int = 1
  act: Callable = nn.tanh

  @nn.compact
  def __call__(self, x, **_):
    layers = [nn.Dense(self.hidden_size)]

    for _ in range(self.n_layers - 1):
      layers += [self.act, nn.Dense(self.hidden_size)]

    layers += [self.act, nn.Dense(self.num_classes)]

    return nn.Sequential(layers)(x)


@register_model
def mlp200(**kwargs):
    MLP200 = partial(TinyMLP, hidden_size=200)
    return MLP200(**kwargs)


@register_model
def mlp16_2(**kwargs):
    MLP16_2 = partial(TinyMLP, hidden_size=16, n_layers=2)
    return MLP16_2(**kwargs)



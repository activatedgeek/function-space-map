from functools import partial
from flax import linen as nn


class TinyMLP(nn.Module):
  num_classes: int
  hidden_size: int

  @nn.compact
  def __call__(self, x, **_):
    x = nn.Dense(self.hidden_size)(x)
    x = nn.tanh(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.tanh(x)

    self.sow('intermediates', 'features', x)

    x = nn.Dense(self.num_classes)(x)
    return x


MLP200 = partial(TinyMLP, hidden_size=200)

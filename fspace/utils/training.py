import logging
from tqdm.auto import tqdm
import jax.numpy as jnp
import flax
from flax.training import train_state
import jax

from .metrics import accuracy, categorical_nll


## Override for extra state variables.
class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict = None

    @property
    def extra_vars(self):
        return {
            v: getattr(self, v)
            for v in ['batch_stats']
            if isinstance(getattr(self, v), flax.core.FrozenDict)
        }


def train_model(state, loader, step_fn, log_dir=None, epoch=None):
    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.numpy(), Y.numpy()

        state, loss = step_fn(state, X, Y)

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item() }
            logging.info(metrics, extra=dict(metrics=True, prefix='sgd/train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


def eval_classifier(state, loader):
    all_logits = []
    all_Y = []

    @jax.jit
    def _forward(X):
        return state.apply_fn({ 'params': state.params, **state.extra_vars}, X,
                              mutable=False, train=False)

    for X, Y in tqdm(loader, leave=False):
        X, Y = X.numpy(), Y.numpy()

        all_logits.append(_forward(X))
        all_Y.append(Y)
    
    all_logits = jnp.concatenate(all_logits)
    all_Y = jnp.concatenate(all_Y)

    acc = accuracy(all_logits, all_Y)
    
    all_nll = categorical_nll(all_logits, all_Y)
    avg_nll = jnp.mean(all_nll, axis=0)

    return {
        'acc': acc.item(),
        'avg_nll': avg_nll.item(),
    }

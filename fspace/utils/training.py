import logging
from tqdm.auto import tqdm
import jax.numpy as jnp
import flax
from flax.training import train_state


## Override to have batch statistics.
class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict


def train_model(state, loader, step_fn, log_dir=None, epoch=None):
    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.numpy(), Y.numpy()

        state, loss = step_fn(state, X, Y)

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item() }
            logging.info(metrics, extra=dict(metrics=True, prefix='sgd/train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


def eval_model(state, loader, step_fn):
    N = len(loader.dataset)
    N_acc = 0
    nll = 0.

    for X, Y in tqdm(loader, leave=False):
        X, Y = X.numpy(), Y.numpy()

        _logits, _nll = step_fn(state, X, Y)
        pred_Y = jnp.argmax(_logits, axis=-1)
        N_acc += jnp.sum(pred_Y == Y)
        nll += _nll

    return { 'acc': N_acc / N, 'nll': nll, 'avg_nll': nll / N }

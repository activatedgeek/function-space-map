import logging
from tqdm.auto import tqdm
import jax.numpy as jnp
import flax
from flax.training import train_state
import jax

from  .third_party.calibration import calibration
from .metrics import accuracy, categorical_nll, categorical_entropy


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

    all_p = jax.nn.softmax(all_logits, axis=-1)

    all_ent = categorical_entropy(all_p)
    avg_ent = jnp.mean(all_ent, axis=0)
    std_ent = jnp.std(all_ent, axis=0)

    ## TODO: JIT this?
    ece, _ = calibration(jax.nn.one_hot(all_Y, loader.dataset.n_classes), all_p, num_bins=10)

    return {
        'acc': acc,
        'avg_nll': avg_nll,
        'avg_ent': avg_ent,
        'std_ent': std_ent,
        'ece': ece,
    }

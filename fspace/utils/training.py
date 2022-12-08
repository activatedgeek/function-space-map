import logging
from tqdm.auto import tqdm
import jax.numpy as jnp
import flax
from flax.training import train_state
import jax
from scipy import stats

from  .third_party.calibration import calibration


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


def eval_model(state, loader, step_fn):
    N = len(loader.dataset)
    N_acc = 0
    nll = 0.
    all_logits = []
    all_Y = []

    for X, Y in tqdm(loader, leave=False):
        X, Y = X.numpy(), Y.numpy()

        _logits, _nll = step_fn(state, X, Y)
        pred_Y = jnp.argmax(_logits, axis=-1)
        
        N_acc += jnp.sum(pred_Y == Y)
        nll += _nll

        all_logits.append(_logits)
        all_Y.append(Y)
    
    all_logits = jnp.concatenate(all_logits)
    all_Y = jnp.concatenate(all_Y)
    all_p = jax.nn.softmax(all_logits, axis=-1)
    
    all_ent = stats.multinomial(1, all_p).entropy()
    avg_ent = jnp.nanmean(all_ent, axis=0)
    std_ent = jnp.nanstd(all_ent, axis=0)
    ece, _ = calibration(jax.nn.one_hot(all_Y, loader.dataset.n_classes), all_p, num_bins=10)

    return {
        'acc': N_acc / N,
        'nll': nll,
        'avg_nll': nll / N,
        'avg_ent': avg_ent,
        'std_ent': std_ent,
        'ece': ece,
    }

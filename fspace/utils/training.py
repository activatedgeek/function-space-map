import logging
from tqdm.auto import tqdm
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax


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

    def apply_gradients(self, *, grads, loss, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, extra_args={ 'loss': loss })
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

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

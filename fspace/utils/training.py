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
    
    @classmethod
    def create(cls, *, apply_fn, params, tx=None, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params) if tx is not None else None
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


def train_model(state, loader, step_fn, log_dir=None, epoch=None):
    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.numpy(), Y.numpy()

        state, step_metrics = step_fn(state, X, Y)

        if log_dir is not None and i % 100 == 0:
            step_metrics = { k: v.item() for k, v in step_metrics.items() }
            logging.info({ 'epoch': epoch, **step_metrics }, extra=dict(metrics=True, prefix='train'))

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


@jax.jit
def update_mutables_fn(state, b_X):
    def fwd_fn(params, **extra_vars):
        _, new_state = state.apply_fn({ 'params': params, **extra_vars }, b_X,
                                        mutable=['batch_stats'], train=True)
        return new_state

    new_state = fwd_fn(state.params, **state.extra_vars)

    final_state = state.replace(**new_state)

    return final_state


def update_mutables(state, loader):
    for X, _ in tqdm(loader, leave=False):
        state = update_mutables_fn(state, X.numpy())
    return state

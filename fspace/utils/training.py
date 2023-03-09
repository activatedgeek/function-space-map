import logging
from tqdm.auto import tqdm
import flax
from flax.training import train_state


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

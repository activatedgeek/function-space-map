import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
import optax

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset
from fspace.nn import create_model
from fspace.utils.training import TrainState, eval_model


@jax.jit
def eval_step_fn(state, b_X, b_Y):
    logits = state.apply_fn({ 'params': state.params, **state.extra_vars}, b_X,
                            mutable=False, train=False)

    nll = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))

    return logits, nll


def train_model(rng, state, loader, ctx_loader, step_fn, log_dir=None, epoch=None):
    ctx_iter = iter(ctx_loader)

    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        try:
            X_ctx, _ = next(ctx_iter)
        except StopIteration:
            ctx_iter = iter(ctx_loader)
            X_ctx, _ = next(ctx_iter)

        # rng_trees = []
        # for _ in range(n_samples):
        #     rng, _tree = tree_split(rng, state.params)
        #     rng_trees.append(_tree)

        state, loss = step_fn(rng, state, X.numpy(), Y.numpy(), X_ctx.numpy())

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item() }
            logging.info(metrics, extra=dict(metrics=True, prefix='sgd/train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, train_subset=1., label_noise=0.,
         batch_size=128, num_workers=4):
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'train_subset': train_subset,
        'label_noise': label_noise,
        'batch_size': batch_size,
    })

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(
        dataset, root=data_dir, seed=seed, train_subset=train_subset, label_noise=label_noise)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    model = create_model(model_name, num_classes=train_data.n_classes)
    # rng, model_init_rng = jax.random.split(rng)

    # init_vars = model.init(model_init_rng, train_data[0][0].numpy()[None, ...])
    init_vars = freeze(checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None))
    logging.info(f'Loaded checkpoint from "{ckpt_path}".')

    other_vars, params = init_vars.pop('params')

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        **other_vars,
        tx=optax.adamw(learning_rate=0.))

    eval_fn = lambda *args: eval_model(*args, eval_step_fn)

    train_metrics = eval_fn(train_state, train_loader)
    logging.info(train_metrics, extra=dict(metrics=True, prefix='train'))

    val_metrics = eval_fn(train_state, val_loader if val_loader.dataset is not None else test_loader)
    logging.info(val_metrics, extra=dict(metrics=True, prefix='val'))

    test_metrics = eval_fn(train_state, test_loader)
    logging.info(test_metrics, extra=dict(metrics=True, prefix='test'))


def entrypoint(log_dir=None, **kwargs):
    log_dir = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

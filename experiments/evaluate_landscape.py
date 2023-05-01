import logging
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from fspace.nn import create_model
from fspace.utils.logging import set_logging, wandb
from fspace.datasets import get_dataset


@partial(jax.jit, static_argnames=['std', 'step_lim', 'n_steps'])
def perturb_params(key, params, std=1., step_lim=1., n_steps=10):

    def _sample(key, p):
        flat_p, tree_unflatten_fn = jax.flatten_util.ravel_pytree(p)
        rv = std * jax.random.normal(key, shape=flat_p.shape, dtype=flat_p.dtype)
        rv = rv / jnp.linalg.norm(rv)
        return tree_unflatten_fn(rv)
    
    step_sizes = jnp.linspace(-step_lim, step_lim, 1 + n_steps) ## additional 1 gets us 0. when n_steps is even, i.e. no perturbation.
    direction = _sample(key, params)

    def _perturb(alpha):
        return jax.tree_util.tree_map(lambda p, d: p + alpha * d, params, direction)
    new_params = jax.vmap(_perturb)(step_sizes)
    
    return new_params


def compute_loss_fn(model, batch_params, batch_extra_vars):
    """Returns function that computes loss data loader.
    """
    @jax.jit
    def model_fn(params, extra_vars, X):
        return model.apply({ 'params': params, **extra_vars }, X, mutable=False, train=False)
    pmap_model_fn = jax.pmap(jax.vmap(model_fn, in_axes=(0, 0, None)), in_axes=(0, 0, None))
    
    @jax.jit
    def loss_fn(logits, Y):
        return optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    pmap_loss_fn = jax.pmap(jax.vmap(loss_fn, in_axes=(0, None)), in_axes=(0, None))
    
    def compute_loss(loader):
        loss_list = []
        
        for X, Y in tqdm(loader, leave=False):
            X, Y = X.numpy(), Y.numpy().astype(int)
            logits = pmap_model_fn(batch_params, batch_extra_vars, X)
            loss = pmap_loss_fn(logits, Y)

            loss_list.append(loss)

        loss_list = jnp.concatenate(loss_list, axis=-1)

        return jnp.mean(loss_list, axis=-1)

    return compute_loss


def compute_mutables_fn(model, batch_params, extra_vars,
                        update_mutables=True):
    @jax.jit
    def model_fn(params, X):
        if update_mutables:
            return model.apply({ 'params': params, **extra_vars }, X, mutable=['batch_stats'], train=True)[-1]
        return extra_vars
    pmap_model_fn = jax.pmap(jax.vmap(model_fn, in_axes=(0, None)), in_axes=(0, None))

    def compute_mutables(loader):
        for X, _ in tqdm(loader, leave=False):
            batch_extra_vars = pmap_model_fn(batch_params, X.numpy())
        return batch_extra_vars
    
    return compute_mutables


def main(seed=None, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, update_mutables=False,
         batch_size=128, num_workers=4,
         step_lim=20., n_directions=1, n_steps=20):
    assert ckpt_path is not None, "Missing checkpoint path."

    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'update_mutables': bool(update_mutables),
        'batch_size': batch_size,
        'step_lim': step_lim,
        'n_directions': n_directions,
        'n_steps': n_steps,
    })

    train_data, *_ = get_dataset(dataset, augment=False, root=data_dir, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

    model, params, extra_vars = create_model(None, model_name, train_data[0][0].numpy()[None, ...],
                                             num_classes=train_data.n_classes, ckpt_path=ckpt_path)

    ## Remove extraneous keys.
    for k in extra_vars.keys():
        if k not in ['batch_stats']:
            extra_vars, _ = extra_vars.pop(k)

    rng = jax.random.PRNGKey(seed)
    
    ## n_directions x n_steps x ... where n_directions are pmap-ed and n_steps are vmap-ed.
    rng, *directions_rng = jax.random.split(rng, 1 + n_directions)
    batch_params = jax.pmap(lambda _k: perturb_params(_k, params, step_lim=step_lim, n_steps=n_steps))(jnp.array(directions_rng))
    batch_extra_vars = compute_mutables_fn(model, batch_params, extra_vars,
                                           update_mutables=bool(update_mutables))(train_loader)
    rnd_directions_loss = compute_loss_fn(model, batch_params, batch_extra_vars)(train_loader)

    if jax.process_index() == 0:
        with open(Path(log_dir) / f'results.npz', 'wb') as f:
            jnp.savez(f,
                      seed=seed,
                      step_lim=step_lim,
                      losses=rnd_directions_loss)

        logging.debug(rnd_directions_loss)
        logging.info(f'Results shape: {rnd_directions_loss.shape}')


def entrypoint(log_dir=None, **kwargs):
    # jax.distributed.initialize()

    log_dir = None
    if jax.process_index() == 0:
        log_dir, finish_logging = set_logging(log_dir=log_dir)

    n_directions = jax.local_device_count()
    main(**kwargs, n_directions=n_directions, log_dir=log_dir)

    if jax.process_index() == 0:
        finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

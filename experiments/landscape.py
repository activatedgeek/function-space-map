from torch.utils.data import DataLoader
from functools import partial
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from fspace.nn import create_model
from fspace.utils.logging import set_logging, wandb
from fspace.datasets import get_dataset


@partial(jax.jit, static_argnames=['std', 'step_lim', 'n_steps'])
def perturb_params(key, params, std=1., step_lim=10., n_steps=10):

    def _sample(key, p):
        flat_p, tree_unflatten_fn = jax.flatten_util.ravel_pytree(p)
        rv = std * jax.random.normal(key, shape=flat_p.shape, dtype=flat_p.dtype)
        rv = rv / jnp.linalg.norm(rv)
        return tree_unflatten_fn(rv)
    
    step_sizes = jnp.linspace(-step_lim, step_lim, n_steps + 1)
    direction = _sample(key, params)

    def _perturb(alpha):
        return jax.tree_util.tree_map(lambda p, d: p + alpha * d, params, direction)
    new_params = jax.vmap(_perturb)(step_sizes)
    
    return new_params


def compute_loss_fn(model, batch_step_params, extra_vars):
    """Returns function that computes loss data loader.
    
    Arguments:
        batch_step_params: n_samples x n_step_sizes x ...
    """

    @jax.jit
    def model_fn(params, X):
        return model.apply({ 'params': params, **extra_vars }, X, mutable=False, train=False)
    
    @jax.jit
    def loss_fn(logits, Y):
        return optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    
    vmap_model_fn = jax.vmap(jax.vmap(model_fn, in_axes=(0, None)), in_axes=(0, None))
    vmap_loss_fn = jax.vmap(jax.vmap(loss_fn, in_axes=(0, None)), in_axes=(0, None))
    
    def compute_loss(loader):
        all_loss = []
        
        for X, Y in tqdm(loader, leave=False):
            X, Y = X.numpy(), Y.numpy()
            logits = vmap_model_fn(batch_step_params, X)
            loss = vmap_loss_fn(logits, Y)
            all_loss.append(loss)

        all_loss = jnp.mean(jnp.concatenate(all_loss, axis=-1), axis=-1)
    
        return all_loss

    return compute_loss


def main(seed=None, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None,
         batch_size=512, num_workers=4,
         n_directions=1, n_steps=10):
    assert ckpt_path is not None, "Missing checkpoint path."

    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'batch_size': batch_size,
        'n_directions': n_directions,
        'n_steps': n_steps,
    })

    _, val_data, _ = get_dataset(dataset, root=data_dir, seed=seed)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers) if val_data is not None else None

    _, model, params, extra_vars = create_model(None, model_name, val_data[0][0].numpy()[None, ...],
                                                num_classes=val_data.n_classes, ckpt_path=ckpt_path)

    rng = jax.random.PRNGKey(seed)
    rng, *samples_rng = jax.random.split(rng, 1 + n_directions)
    
    rnd_params = jax.vmap(lambda _k: perturb_params(_k, params, n_steps=n_steps))(jnp.array(samples_rng))
    
    compute_loss = compute_loss_fn(model, rnd_params, extra_vars)
    rnd_directions_loss = compute_loss(val_loader)  ## n_directions x n_steps

    breakpoint()


def entrypoint(log_dir=None, **kwargs):
    log_dir, finish_logging = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

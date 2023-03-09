from typing import Tuple
from functools import partial
import chex
import jax
import jax.numpy as jnp
import optax

from .state import SWAGDiagState, SWAGState


@jax.jit
def tree_random_split(key: chex.PRNGKey,
                      ref_tree: chex.ArrayTree) -> Tuple[chex.PRNGKey, chex.ArrayTree]:

    treedef = jax.tree_util.tree_structure(ref_tree)

    key, *key_list = jax.random.split(key, 1 + treedef.num_leaves)

    return key, jax.tree_util.tree_unflatten(treedef, key_list)


@jax.jit
def tree_random_normal(key: chex.PRNGKey,
                       ref_tree: chex.ArrayTree) -> chex.ArrayTree:
    
    _, key_tree = tree_random_split(key, ref_tree)

    return jax.tree_util.tree_map(lambda k, v: jax.random.normal(k, v.shape, v.dtype),
                                  key_tree, ref_tree)


@jax.jit
def sample_swag_diag(key: chex.PRNGKey, state: SWAGDiagState, eps: float = 1e-30) -> optax.Params:
    mean_tree = state.mean
    std_tree = jax.tree_util.tree_map(lambda mu, p2: jnp.sqrt(jnp.clip(p2 - jnp.square(mu), a_min=eps)),
                                      mean_tree, state.params2)
    
    z_tree = tree_random_normal(key, mean_tree)

    return jax.tree_util.tree_map(lambda mu, std, z: mu + std * z, mean_tree, std_tree, z_tree)


@partial(jax.jit, static_argnames=['rank'])
def sample_swag(key: chex.PRNGKey, state: SWAGState, rank: int, scale: float = 1., eps: float = 1e-30) -> optax.Params:
    mean, tree_unflatten_fn = jax.flatten_util.ravel_pytree(state.mean)
    p2, _ = jax.flatten_util.ravel_pytree(state.params2)
    
    std = jnp.sqrt(jnp.clip(p2 - jnp.square(mean), a_min=eps))

    dparams = jax.vmap(lambda tree: jax.flatten_util.ravel_pytree(tree)[0])(state.dparams)

    z1_key, z2_key = jax.random.split(key, 2)

    z1 = jax.random.normal(z1_key, mean.shape)
    z1_scale = scale / jnp.sqrt(2)

    z2 = jax.random.normal(z2_key, (rank,))
    z2_scale = scale / jnp.sqrt(2 * (rank - 1))

    return tree_unflatten_fn(mean + z1_scale * std * z1 + z2_scale * jnp.matmul(dparams.T, z2))

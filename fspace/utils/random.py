from typing import Tuple
import chex
import jax
import jax.numpy as jnp


@jax.jit
def tree_split(key: chex.PRNGKey,
               ref_tree: chex.ArrayTree) -> Tuple[chex.PRNGKey, chex.ArrayTree]:

    treedef = jax.tree_util.tree_structure(ref_tree)

    key, *key_list = jax.random.split(key, 1 + treedef.num_leaves)

    return key, jax.tree_util.tree_unflatten(treedef, key_list)


@jax.jit
def sample_tree_diag_gaussian(key: chex.PRNGKey, 
                              mean_tree: chex.ArrayTree, var_tree: chex.ArrayTree) -> chex.ArrayTree:
    
    _, key_tree = tree_split(key, mean_tree)
    
    def _sample_param(key: chex.PRNGKey,
                      mu: chex.Array, var: chex.Array) -> chex.Array:
        return mu + jnp.sqrt(var) * jax.random.normal(key, mu.shape, mu.dtype)
    
    return jax.tree_util.tree_map(_sample_param, key_tree, mean_tree, var_tree)

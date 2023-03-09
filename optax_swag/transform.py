from typing import NamedTuple, Tuple
import chex
import optax
import jax.numpy as jnp
import jax


class SWAGState(NamedTuple):
  """State for SWAG mean and non-centered variance."""
  k: chex.Array  # Step count.
  n: chex.Array  # Iterate count using for running stats.
  mean: optax.Params  # Running mean of iterates.
  params2: optax.Params  # Running non-centered variance of iterates.


def swag_diag(update_freq: int) -> optax.GradientTransformation:

  def init_fn(params: optax.Params) -> SWAGState:
    return SWAGState(
        k=jnp.zeros([], jnp.int32),
        n=jnp.zeros([], jnp.int32),
        mean=params,
        params2=jax.tree_util.tree_map(lambda t: jnp.square(t), params))

  def update_fn(updates: optax.Updates, state: SWAGState,
                params: optax.Params) -> Tuple[optax.Updates, SWAGState]:
    
    k = (state.k + 1) % update_freq
    update_mask = k == 0
    n = state.n + 1 * update_mask

    next_params = jax.tree_util.tree_map(lambda u, p: u * update_mask + p, updates, params)
    next_mean = jax.tree_util.tree_map(lambda mu, np: (n * mu + np) / (n + 1), 
                                       state.mean, next_params)
    next_params2 = jax.tree_util.tree_map(lambda v, np: (n * v + jnp.square(np)) / (n + 1),
                                          state.params2, next_params)

    return updates, SWAGState(k=k, n=n, mean=next_mean, params2=next_params2)

  return optax.GradientTransformation(init_fn, update_fn)
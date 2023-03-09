import logging
from typing import Tuple
import optax
import jax.numpy as jnp
import jax

from .state import SWAState, SWAGDiagState, SWAGState


def swa(freq: int) -> optax.GradientTransformation:
  assert freq > 0, 'freq must be positive integer.'

  def init_fn(params: optax.Params) -> SWAState:
    return SWAState(mean=params)

  def update_fn(updates: optax.Updates, state: SWAState,
                params: optax.Params) -> Tuple[optax.Updates, SWAState]:
    
    next_step = (state.step + 1) % freq
    update_mask = next_step == 0
    n = state.n + 1 * update_mask

    next_params = jax.tree_util.tree_map(lambda p, u: jnp.where(update_mask, p + u, p),
                                         params, updates)
    next_mean = jax.tree_util.tree_map(lambda mu, np: jnp.where(update_mask, (n * mu + np) / (n + 1), mu),
                                       state.mean, next_params)

    return updates, SWAState(step=next_step, n=n, mean=next_mean)

  return optax.GradientTransformation(init_fn, update_fn)


def swag_diag(freq: int) -> optax.GradientTransformation:
  assert freq > 0, 'freq must be positive integer.'

  def init_fn(params: optax.Params) -> SWAGDiagState:
    return SWAGDiagState(
        mean=params,
        params2=jax.tree_util.tree_map(lambda p: jnp.square(p), params))

  def update_fn(updates: optax.Updates, state: SWAGDiagState,
                params: optax.Params) -> Tuple[optax.Updates, SWAGDiagState]:
    
    next_step = (state.step + 1) % freq
    update_mask = next_step == 0
    n = state.n + 1 * update_mask

    next_params = jax.tree_util.tree_map(lambda p, u: jnp.where(update_mask, p + u, p),
                                         params, updates)
    next_mean = jax.tree_util.tree_map(lambda mu, np: jnp.where(update_mask, (n * mu + np) / (n + 1), mu),
                                       state.mean, next_params)
    next_params2 = jax.tree_util.tree_map(lambda p2, np: jnp.where(update_mask, (n * p2 + jnp.square(np)) / (n + 1), p2),
                                          state.params2, next_params)

    return updates, SWAGDiagState(step=next_step, n=n, mean=next_mean, params2=next_params2)

  return optax.GradientTransformation(init_fn, update_fn)


def swag(freq: int, rank: int) -> optax.GradientTransformation:
  assert freq > 0, 'freq must be positive integer.'
  if rank < 2:
    logging.warning('Rank must be greater than 1. Switching to swag_diag.')
    return swag_diag(freq)

  def init_fn(params: optax.Params) -> SWAGState:
    return SWAGState(
        mean=params,
        params2=jax.tree_util.tree_map(lambda p: jnp.square(p), params),
        dparams=jax.tree_util.tree_map(lambda p: jnp.zeros_like(jnp.repeat(p[jnp.newaxis, ...], rank, axis=0)), params))

  def update_fn(updates: optax.Updates, state: SWAGState,
                params: optax.Params) -> Tuple[optax.Updates, SWAGState]:
    
    next_step = (state.step + 1) % freq
    update_mask = next_step == 0
    n = state.n + 1 * update_mask

    next_params = jax.tree_util.tree_map(lambda p, u: jnp.where(update_mask, p + u, p),
                                         params, updates)
    next_mean = jax.tree_util.tree_map(lambda mu, np: jnp.where(update_mask, (n * mu + np) / (n + 1), mu),
                                       state.mean, next_params)
    next_params2 = jax.tree_util.tree_map(lambda p2, np: jnp.where(update_mask, (n * p2 + jnp.square(np)) / (n + 1), p2),
                                          state.params2, next_params)
    next_dparams = jax.tree_util.tree_map(lambda dp, np, nmu: jnp.where(update_mask, dp.at[state.c].set(np - nmu), dp),
                                          state.dparams, next_params, next_mean)

    c = (state.c + 1 * update_mask) % rank

    return updates, SWAGState(step=next_step, n=n, c=c, mean=next_mean, params2=next_params2, dparams=next_dparams)

  return optax.GradientTransformation(init_fn, update_fn)
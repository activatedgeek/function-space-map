from typing import NamedTuple
import jax.numpy as jnp
import optax
import chex


class SWAState(NamedTuple):
  """State for SWAG mean and non-centered variance."""
  mean: optax.Params  # Running mean of iterates.
  step: chex.Array = jnp.zeros([], jnp.int32)  # Step count.
  n: chex.Array = jnp.zeros([], jnp.int32)  # Iterate count using for running stats.


class SWAGDiagState(NamedTuple):
  """State for SWAG mean and diagonal non-centered variance."""
  mean: optax.Params  # Running mean of iterates.
  params2: optax.Params  # Running non-centered variance of iterates.
  step: chex.Array = jnp.zeros([], jnp.int32)  # Step count.
  n: chex.Array = jnp.zeros([], jnp.int32)  # Iterate count using for running stats.


class SWAGState(NamedTuple):
  """State for SWAG mean, diagonal non-centered variance and low rank terms."""
  mean: optax.Params  # Running mean of iterates.
  params2: optax.Params  # Running non-centered variance of iterates.
  dparams: optax.Params  # Low rank delta columns.
  step: chex.Array = jnp.zeros([], jnp.int32)  # Step count.
  n: chex.Array = jnp.zeros([], jnp.int32)  # Iterate count using for running stats.
  c: chex.Array = jnp.zeros([], jnp.int32)  # Current column to update.

import jax
import jax.numpy as jnp
import optax


@jax.jit
def accuracy(logits_or_p, Y):
    '''Compute accuracy

    Arguments:
        logits_or_p: (B, d)
        Y: (B,) integer labels.
    '''
    matches = jnp.argmax(logits_or_p, axis=-1) == Y
    return jnp.mean(matches)


@jax.jit
def categorical_nll(logits, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return optax.softmax_cross_entropy_with_integer_labels(logits, Y)


@jax.jit
def categorical_entropy(p):
    '''Entropy of categorical distribution.

    Arguments:
        p: (B, d)
    
    Returns:
        (B,)
    '''
    return - jnp.sum(p * jnp.log(p + 1e-6), axis=-1)

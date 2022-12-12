import jax
import jax.numpy as jnp
import optax
from sklearn.metrics import auc

from .third_party.calibration import calibration  ## For external usage.


@jax.jit
def accuracy(logits_or_p, Y):
    '''Compute accuracy

    Arguments:
        logits_or_p: (B, d)
        Y: (B,) integer labels.
    '''
    if len(Y) == 0:
        return 0.
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


# @jax.jit
def selective_accuracy(p, Y):
    '''Selective Prediction Accuracy

    Uses predictive entropy with T thresholds.

    Arguments:
        p: (B, d)
    
    Returns:
        (B,)
    '''
    T = 10
    d = p.shape[-1]
    thresholds = jnp.atleast_2d(
        jnp.linspace(0., jnp.log(d) + 1e-8, num=T)).T ## (T,1)

    entropy = jnp.atleast_2d(categorical_entropy(p)) ## (1,B)

    sel_mask = entropy < thresholds
    accuracies = [accuracy(p[jnp.nonzero(mask)], Y[jnp.nonzero(mask)]) for mask in sel_mask]

    norm_thresholds = thresholds / jnp.log(d)

    return auc(norm_thresholds, accuracies)

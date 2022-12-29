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

    thresholds = jnp.concatenate([jnp.linspace(100, 1, 100), jnp.array([0.1])], axis=0)

    predictions_test = p.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = jnp.percentile(scores_id, threshold)
        mask = jnp.array(scores_id <= p)
        thresholded_accuracies.append(jnp.mean(accuracies_test[mask]))
    values_id = jnp.array(thresholded_accuracies)

    auc_sel_id = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel_id += (x * values_id[i] + x * values_id[i+1]) / 2

    return auc_sel_id

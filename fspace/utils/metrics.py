import jax
import jax.numpy as jnp
import optax
import distrax
from sklearn.metrics import roc_auc_score

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
def categorical_nll_with_p(p, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return - distrax.Categorical(probs=p).log_prob(Y)


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
def selective_accuracy_auc(p, Y):
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


def entropy_ood_auc(p, ood_p):
    ent = categorical_entropy(p)
    targets = jnp.zeros_like(ent)

    ood_ent = categorical_entropy(ood_p)
    ood_targets = jnp.ones_like(ood_ent)

    all_ent = jnp.concatenate([ent, ood_ent])
    all_targets = jnp.concatenate([targets, ood_targets])
    
    return roc_auc_score(all_targets, all_ent)


def cheap_eval_classifier(all_p, all_Y):
    acc = accuracy(all_p, all_Y)

    avg_nll = jnp.mean(categorical_nll_with_p(all_p, all_Y), axis=0)

    avg_ent = jnp.mean(categorical_entropy(all_p), axis=0)

    return {
        'acc': acc.item(),
        'avg_nll': avg_nll.item(),
        'avg_ent': avg_ent.item(),
    }


def eval_classifier(all_p, all_Y):
    acc = accuracy(all_p, all_Y)

    sel_acc = selective_accuracy_auc(all_p, all_Y)

    avg_nll = jnp.mean(categorical_nll_with_p(all_p, all_Y), axis=0)

    avg_ent = jnp.mean(categorical_entropy(all_p), axis=0)

    try:
        ## TODO: JIT this?
        ece, _ = calibration(jax.nn.one_hot(all_Y, all_p.shape[-1]), all_p, num_bins=10)
    except ZeroDivisionError:
        ece = jnp.array([jnp.nan])

    return {
        'acc': acc.item(),
        'sel_acc': sel_acc.item(),
        'avg_nll': avg_nll.item(),
        'avg_ent': avg_ent.item(),
        'ece': ece.item(),
    }

import logging
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp

from fspace.nn import create_model
from fspace.utils.metrics import \
    accuracy, selective_accuracy_auc, categorical_entropy, categorical_nll, calibration, entropy_ood_auc


def eval_logits(f, loader):
    all_logits = []
    all_Y = []

    for X, Y in tqdm(loader, leave=False):
        X, Y = X.numpy(), Y.numpy()

        all_logits.append(f(X))
        all_Y.append(Y)

    all_logits = jnp.concatenate(all_logits)
    all_Y = jnp.concatenate(all_Y)

    return all_logits, all_Y


def eval_classifier(all_logits, all_Y):
    n_classes = all_logits.shape[-1]

    all_p = jax.nn.softmax(all_logits, axis=-1)

    acc = accuracy(all_logits, all_Y)
    sel_acc = selective_accuracy_auc(all_p, all_Y)

    all_nll = categorical_nll(all_logits, all_Y)
    avg_nll = jnp.mean(all_nll, axis=0)

    all_ent = categorical_entropy(all_p)
    avg_ent = jnp.mean(all_ent, axis=0)

    ## TODO: JIT this?
    ece, _ = calibration(jax.nn.one_hot(all_Y, n_classes), all_p, num_bins=10)

    return {
        'acc': acc.item(),
        'sel_acc': sel_acc.item(),
        'avg_nll': avg_nll.item(),
        'avg_ent': avg_ent.item(),
        'ece': ece.item(),
    }


def full_eval_model(model_name, num_classes, ckpt_path,
                    train_loader, test_loader, val_loader=None, ood_loader=None,
                    ckpt_prefix='checkpoint_', log_prefix=None):
    _, model, params, other_vars = create_model(None, model_name, None, num_classes=num_classes,
                                                ckpt_path=ckpt_path, ckpt_prefix=ckpt_prefix)

    @jax.jit
    def f_model(X):
        return model.apply({ 'params': params, **other_vars}, X, mutable=False, train=False)

    logging.info(f'Evaluating train metrics...')
    train_metrics = eval_classifier(*eval_logits(f_model, train_loader))
    logging.info(train_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}train'))

    logging.info(f'Evaluating test metrics...')
    test_logits, test_Y = eval_logits(f_model, test_loader)
    test_metrics = eval_classifier(test_logits, test_Y)
    logging.info(test_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}test'))

    if val_loader is not None:
        logging.info(f'Evaluating validation metrics...')
        val_metrics = eval_classifier(*eval_logits(f_model, val_loader))
    else:
        val_metrics = test_metrics
    logging.info(val_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}val'))

    if ood_loader is not None:
        ood_test_logits, ood_test_Y = eval_logits(f_model, ood_loader)
        ood_test_metrics = eval_classifier(ood_test_logits, ood_test_Y)
        ood_auc = entropy_ood_auc(test_logits, ood_test_logits)

        logging.info({ **ood_test_metrics, 'auc': ood_auc }, extra=dict(metrics=True, prefix=f'{log_prefix}ood_test'))

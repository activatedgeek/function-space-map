import logging
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp

from fspace.nn import create_model
from fspace.utils.metrics import \
    accuracy, \
    selective_accuracy_auc, \
    categorical_entropy, \
    categorical_nll_with_p, \
    calibration, \
    entropy_ood_auc


def eval_classifier(all_p, all_Y):
    acc = accuracy(all_p, all_Y)

    sel_acc = selective_accuracy_auc(all_p, all_Y)

    avg_nll = jnp.mean(categorical_nll_with_p(all_p, all_Y), axis=0)

    avg_ent = jnp.mean(categorical_entropy(all_p), axis=0)

    ## TODO: JIT this?
    ece, _ = calibration(jax.nn.one_hot(all_Y, all_p.shape[-1]), all_p, num_bins=10)

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

    def compute_model_p(loader):
        @jax.jit
        def model_fn(X):
            return model.apply({ 'params': params, **other_vars }, X, mutable=False, train=False)

        all_logits = []
        all_Y = []

        for X, Y in tqdm(loader, leave=False):
            X, Y = X.numpy(), Y.numpy()

            all_logits.append(model_fn(X))
            all_Y.append(Y)

        all_p = jax.nn.softmax(jnp.concatenate(all_logits), axis=-1)
        all_Y = jnp.concatenate(all_Y)

        return all_p, all_Y

    logging.info(f'Evaluating train metrics...')
    train_p, train_Y = compute_model_p(train_loader)
    train_metrics = eval_classifier(train_p, train_Y)
    logging.info(train_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}train'))

    logging.info(f'Evaluating test metrics...')
    test_p, test_Y = compute_model_p(test_loader)
    test_metrics = eval_classifier(test_p, test_Y)
    logging.info(test_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}test'))

    if val_loader is not None:
        logging.info(f'Evaluating validation metrics...')
        val_p, val_Y = compute_model_p(val_loader)
        val_metrics = eval_classifier(val_p, val_Y)
    else:
        val_metrics = test_metrics
    logging.info(val_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}val'))

    if ood_loader is not None:
        ood_p, ood_Y = compute_model_p(ood_loader)
        ood_test_metrics = eval_classifier(ood_p, ood_Y)
        ood_auc = entropy_ood_auc(test_p, ood_p)

        logging.info({ **ood_test_metrics, 'auc': ood_auc }, extra=dict(metrics=True, prefix=f'{log_prefix}ood_test'))

import logging
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp

from fspace.nn import create_model
from fspace.utils.metrics import eval_classifier, entropy_ood_auc


def compute_p_fn(model_name, num_classes, ckpt_path, ckpt_prefix='checkpoint_'):
    _, model, params, other_vars = create_model(None, model_name, None, num_classes=num_classes,
                                                ckpt_path=ckpt_path, ckpt_prefix=ckpt_prefix)

    @jax.jit
    def model_fn(X):
        return model.apply({ 'params': params, **other_vars }, X, mutable=False, train=False)

    def compute_p(loader):
        all_logits = []
        all_Y = []

        for X, Y in tqdm(loader, leave=False):
            X, Y = X.numpy(), Y.numpy()

            all_logits.append(model_fn(X))
            all_Y.append(Y)

        all_p = jax.nn.softmax(jnp.concatenate(all_logits), axis=-1)
        all_Y = jnp.concatenate(all_Y)

        return all_p, all_Y
    
    return compute_p


def full_eval_model(compute_model_p,
                    train_loader, test_loader, val_loader=None, ood_loader=None,
                    log_prefix=None):

    logging.info(f'Computing train metrics...')
    train_p, train_Y = compute_model_p(train_loader)
    train_metrics = eval_classifier(train_p, train_Y)
    logging.info(train_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}train'))

    logging.info(f'Computing test metrics...')
    test_p, test_Y = compute_model_p(test_loader)
    test_metrics = eval_classifier(test_p, test_Y)
    logging.info(test_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}test'))

    if val_loader is not None:
        logging.info(f'Computing validation metrics...')
        val_p, val_Y = compute_model_p(val_loader)
        val_metrics = eval_classifier(val_p, val_Y)
    else:
        val_metrics = test_metrics
    logging.info(val_metrics, extra=dict(metrics=True, prefix=f'{log_prefix}val'))

    if ood_loader is not None:
        logging.info(f'Computing OOD metrics...')
        ood_p, ood_Y = compute_model_p(ood_loader)
        ood_test_metrics = eval_classifier(ood_p, ood_Y)
        ood_auc = entropy_ood_auc(test_p, ood_p)

        logging.info({ **ood_test_metrics, 'auc': ood_auc }, extra=dict(metrics=True, prefix=f'{log_prefix}ood_test'))

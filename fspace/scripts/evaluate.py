import logging
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp

from fspace.utils.metrics import cheap_eval_classifier, eval_classifier, entropy_ood_auc


def compute_prob_ensemble_fn(model, batch_params, batch_extra_vars):
    """Compute classifier ensemble.

    A `jax.vmap` over the model parameters.
    """

    @jax.jit
    def model_fn(params, extra_vars, X):
        return model.apply(
            {"params": params, **extra_vars}, X, mutable=False, train=False
        )

    ens_model_fn = jax.vmap(model_fn, in_axes=(0, 0, None))

    def compute_ensemble(loader):
        all_ens_logits = []
        all_Y = []

        for X, Y in tqdm(loader, leave=False):
            X, Y = X.numpy(), Y.numpy()
            all_ens_logits.append(ens_model_fn(batch_params, batch_extra_vars, X))
            all_Y.append(Y)

        all_p = jnp.mean(
            jax.nn.softmax(jnp.concatenate(all_ens_logits, axis=-2), axis=-1), axis=0
        )
        all_Y = jnp.concatenate(all_Y)

        return all_p, all_Y

    return compute_ensemble


def compute_prob_fn(model, params, extra_vars):
    """Compute classifier output"""
    batch_params = jax.tree_util.tree_map(lambda p: p[jnp.newaxis, ...], params)
    batch_extra_vars = jax.tree_util.tree_map(lambda e: e[jnp.newaxis, ...], extra_vars)
    return compute_prob_ensemble_fn(model, batch_params, batch_extra_vars)


def compute_mutables_fn(model, batch_params):
    @jax.jit
    def model_fn(params, extra_vars, X):
        return model.apply(
            {"params": params, **extra_vars}, X, mutable=["batch_stats"], train=True
        )[-1]

    ens_model_fn = jax.vmap(model_fn, in_axes=(0, 0, None))

    def compute_mutables(loader, batch_extra_vars):
        for X, _ in tqdm(loader, leave=False):
            batch_extra_vars = ens_model_fn(batch_params, batch_extra_vars, X.numpy())
        return batch_extra_vars

    return compute_mutables


def cheap_eval_model(compute_model_p, loader):
    p, Y = compute_model_p(loader)
    return cheap_eval_classifier(p, Y)


def full_eval_model(
    compute_model_p,
    train_loader,
    test_loader,
    val_loader=None,
    ood_loader=None,
    log_prefix=None,
):
    logging.info(f"Computing train metrics...")

    train_p, train_Y = compute_model_p(train_loader)
    train_metrics = eval_classifier(train_p, train_Y)

    logging.info(train_metrics, extra=dict(metrics=True, prefix=f"{log_prefix}train"))
    logging.debug(train_metrics)

    logging.info(f"Computing test metrics...")

    test_p, test_Y = compute_model_p(test_loader)
    test_metrics = eval_classifier(test_p, test_Y)

    logging.info(test_metrics, extra=dict(metrics=True, prefix=f"{log_prefix}test"))
    logging.debug(test_metrics)

    if val_loader is not None:
        logging.info(f"Computing validation metrics...")

        val_p, val_Y = compute_model_p(val_loader)
        val_metrics = eval_classifier(val_p, val_Y)

        logging.info(val_metrics, extra=dict(metrics=True, prefix=f"{log_prefix}val"))
        logging.debug(val_metrics)

    if ood_loader is not None:
        logging.info(f"Computing OOD metrics...")

        ood_p, ood_Y = compute_model_p(ood_loader)
        ood_test_metrics = eval_classifier(ood_p, ood_Y)
        ood_auc = entropy_ood_auc(test_p, ood_p)
        ood_metrics = {**ood_test_metrics, "auc": ood_auc}

        logging.info(
            ood_metrics, extra=dict(metrics=True, prefix=f"{log_prefix}ood_test")
        )
        logging.debug(ood_metrics)

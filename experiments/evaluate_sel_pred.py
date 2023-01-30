import logging
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader

from fspace.utils.logging import set_logging, wandb
from fspace.nn import create_model
from fspace.scripts.evaluate import eval_logits
from fspace.datasets import get_dataset
from fspace.utils.metrics import categorical_entropy


def selective_accuracy(p, Y):
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

    return thresholds, values_id


def main(log_dir=None, model_name=None, dataset=None, corr_config=None, ckpt_path=None):
    wandb.config.update({
        'log_dir': log_dir,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'corr_config': corr_config,  # CIFAR-10 corruption config name.
    })

    _, _, test_data = get_dataset(dataset, corr_config=corr_config)
    test_loader = DataLoader(test_data, batch_size=128, num_workers=4)

    _, model, params, other_vars = create_model(None, model_name, None, num_classes=10,
                                                ckpt_path=ckpt_path, ckpt_prefix='checkpoint_')

    @jax.jit
    def f_model(X):
        return model.apply({ 'params': params, **other_vars }, X, mutable=False, train=False)

    all_logits, all_Y = eval_logits(f_model, test_loader)

    thresholds, sel_preds = selective_accuracy(jax.nn.softmax(all_logits, axis=-1), all_Y)

    for i, (t, s) in enumerate(zip(thresholds.tolist(), sel_preds.tolist())):
        logging.info({ 'x_id': i, 'threshold': 100 - t, 'sel_pred': s }, extra=dict(metrics=True, prefix='s/test'))


def entrypoint(log_dir=None, **kwargs):
    log_dir, finish_logging = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

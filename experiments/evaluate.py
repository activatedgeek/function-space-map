import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
import jax
import jax.numpy as jnp

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
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


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ood_dataset=None,
         batch_size=512, num_workers=4):
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'ood_dataset': ood_dataset,
        'batch_size': batch_size,
    })

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(dataset, root=data_dir, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    model = create_model(model_name, num_classes=train_data.n_classes)

    if ckpt_path is not None:
        init_vars = freeze(checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None))
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_vars = model.init(model_init_rng, train_data[0][0].numpy()[None, ...])
        logging.warning(f'Initialized a random model.')

    other_vars, params = init_vars.pop('params')

    @jax.jit
    def f_model(X):
        return model.apply({ 'params': params, **other_vars}, X, mutable=False, train=False)

    logging.info(f'Evaluating train metrics...')
    train_metrics = eval_classifier(*eval_logits(f_model, train_loader))
    logging.info(train_metrics, extra=dict(metrics=True, prefix='train'))

    logging.info(f'Evaluating test metrics...')
    test_logits, test_Y = eval_logits(f_model, test_loader)
    test_metrics = eval_classifier(test_logits, test_Y)
    logging.info(test_metrics, extra=dict(metrics=True, prefix='test'))

    if val_loader.dataset is not None:
        logging.info(f'Evaluating validation metrics...')
        val_metrics = eval_classifier(*eval_logits(f_model, val_loader))
    else:
        val_metrics = test_metrics
    logging.info(val_metrics, extra=dict(metrics=True, prefix='val'))

    if ood_dataset is not None:
        logging.info(f'Evaluating OOD metrics...')

        _, _, ood_test_data = get_dataset(ood_dataset, root=data_dir, seed=seed,
                                          normalize=get_dataset_normalization(dataset))
        ood_test_loader = DataLoader(ood_test_data, batch_size=batch_size, num_workers=num_workers)

        ood_test_logits, ood_test_Y = eval_logits(f_model, ood_test_loader)
        ood_test_metrics = eval_classifier(ood_test_logits, ood_test_Y)
        ood_auc = entropy_ood_auc(test_logits, ood_test_logits)

        logging.info({ **ood_test_metrics, 'auc': ood_auc }, extra=dict(metrics=True, prefix='ood_test'))


def entrypoint(log_dir=None, **kwargs):
    log_dir = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

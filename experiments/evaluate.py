import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
import jax
import jax.numpy as jnp
import optax

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset
from fspace.nn import create_model
from fspace.utils.training import TrainState
from fspace.utils.metrics import \
    accuracy, selective_accuracy, categorical_entropy, categorical_nll, calibration


def eval_classifier(state, loader):
    all_logits = []
    all_Y = []

    @jax.jit
    def _forward(X):
        return state.apply_fn({ 'params': state.params, **state.extra_vars}, X,
                              mutable=False, train=False)

    for X, Y in tqdm(loader, leave=False):
        X, Y = X.numpy(), Y.numpy()

        all_logits.append(_forward(X))
        all_Y.append(Y)

    all_logits = jnp.concatenate(all_logits)
    all_Y = jnp.concatenate(all_Y)

    all_p = jax.nn.softmax(all_logits, axis=-1)

    acc = accuracy(all_logits, all_Y)
    sel_acc = selective_accuracy(all_p, all_Y)

    all_nll = categorical_nll(all_logits, all_Y)
    avg_nll = jnp.mean(all_nll, axis=0)

    all_ent = categorical_entropy(all_p)
    avg_ent = jnp.mean(all_ent, axis=0)

    ## TODO: JIT this?
    ece, _ = calibration(jax.nn.one_hot(all_Y, loader.dataset.n_classes), all_p, num_bins=10)

    return {
        'acc': acc.item(),
        'sel_acc': sel_acc.item(),
        'avg_nll': avg_nll.item(),
        'avg_ent': avg_ent.item(),
        'ece': ece.item(),
    }


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None,
         batch_size=128, num_workers=4):
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
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

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        **other_vars,
        tx=optax.sgd(learning_rate=0.))

    train_metrics = eval_classifier(train_state, train_loader)
    logging.info(train_metrics, extra=dict(metrics=True, prefix='train'))

    val_metrics = eval_classifier(train_state, val_loader if val_loader.dataset is not None else test_loader)
    logging.info(val_metrics, extra=dict(metrics=True, prefix='val'))

    test_metrics = eval_classifier(train_state, test_loader)
    logging.info(test_metrics, extra=dict(metrics=True, prefix='test'))


def entrypoint(log_dir=None, **kwargs):
    log_dir = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

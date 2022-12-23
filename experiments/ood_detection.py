import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
import optax
from sklearn.metrics import roc_auc_score

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
from fspace.nn import create_model
from fspace.utils.training import TrainState
from fspace.utils.metrics import categorical_entropy


def eval_classifier_entropy(state, loader):
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

    all_ent = categorical_entropy(all_p)

    return all_ent


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ood_dataset=None,
         batch_size=256, num_workers=4):
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

    _, _, test_data = get_dataset(dataset, root=data_dir, seed=seed)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    _, _, ood_test_data = get_dataset(ood_dataset, root=data_dir, seed=seed,
                                      normalize=get_dataset_normalization(dataset))
    ood_test_loader = DataLoader(ood_test_data, batch_size=batch_size, num_workers=num_workers)

    model = create_model(model_name, num_classes=test_data.n_classes)

    if ckpt_path is not None:
        init_vars = freeze(checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None))
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_vars = model.init(model_init_rng, test_data[0][0].numpy()[None, ...])
        logging.warning(f'Initialized a random model.')

    other_vars, params = init_vars.pop('params')

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        **other_vars,
        tx=optax.sgd(learning_rate=0.))

    p_ent = eval_classifier_entropy(state, test_loader)

    ood_p_ent = eval_classifier_entropy(state, ood_test_loader)

    all_ent = jnp.concatenate([p_ent, ood_p_ent])
    all_targets = jnp.concatenate([jnp.zeros(ood_p_ent.shape[0]), jnp.ones(p_ent.shape[0])])

    auroc = roc_auc_score(all_targets, all_ent)

    logging.info({
        'auroc': auroc,
        'avg_ent': jnp.mean(p_ent, axis=0),
        'ood_avg_ent': jnp.mean(ood_p_ent, axis=0),
    }, extra=dict(metrics=True, prefix='test'))


def entrypoint(log_dir=None, **kwargs):
    log_dir = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

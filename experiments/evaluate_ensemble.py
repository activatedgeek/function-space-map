import logging
from torch.utils.data import DataLoader
import jax.numpy as jnp

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
from fspace.scripts.evaluate import full_eval_model, compute_p_fn


## TODO: jax.vmap over ensemble parameters.
def compute_ensemble_p_fn(model_name, num_classes, ckpt_paths, ckpt_prefix='checkpoint_'):
    ensemble_p_fns = [
        compute_p_fn(model_name, num_classes, ckpt_path, ckpt_prefix=ckpt_prefix)
        for ckpt_path in ckpt_paths
    ]

    def compute_ensemble_p(loader):
        ensemble_p, all_Y = [], None
        for compute_p in ensemble_p_fns:
            all_p, all_Y = compute_p(loader)
            ensemble_p.append(all_p)
        
        ensemble_p = jnp.mean(jnp.stack(ensemble_p), axis=0)

        return ensemble_p, all_Y

    return compute_ensemble_p


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None,
         dataset=None, ood_dataset=None, corr_config=None,
         batch_size=512, num_workers=4):
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'dataset': dataset,
        'corr_config': corr_config,  # CIFAR-10 corruption config name.
        'ood_dataset': ood_dataset,
        'batch_size': batch_size,
    })

    train_data, val_data, test_data = get_dataset(dataset, root=data_dir, seed=seed, corr_config=corr_config)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    ood_test_loader = None
    if ood_dataset is not None:
        _, _, ood_test_data = get_dataset(ood_dataset, root=data_dir, seed=seed,
                                          normalize=get_dataset_normalization(dataset))
        ood_test_loader = DataLoader(ood_test_data, batch_size=batch_size, num_workers=num_workers)


    ckpt_paths = None

    assert ckpt_paths is not None

    logging.info(f'Evaluating latest checkpoint...')
    full_eval_model(compute_ensemble_p_fn(model_name, train_data.n_classes, ckpt_paths, ckpt_prefix='checkpoint_'),
                    train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                    log_prefix='s/')

    try:
        logging.info(f'Evaluating best (validation) checkpoint...')
        full_eval_model(compute_ensemble_p_fn(model_name, train_data.n_classes, ckpt_paths, ckpt_prefix='best_checkpoint_'),
                        train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                        log_prefix='s/best/')
    except TypeError:
        logging.warning('Skipping best checkpoint evaluation.')


def entrypoint(log_dir=None, **kwargs):
    log_dir = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

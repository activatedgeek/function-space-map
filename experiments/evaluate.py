from torch.utils.data import DataLoader

from fspace.nn import create_model
from fspace.utils.logging import set_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
from fspace.scripts.evaluate import full_eval_model, compute_prob_fn


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ood_dataset=None, corr_config=None,
         batch_size=512, num_workers=4):
    assert ckpt_path is not None, "Missing checkpoint path."

    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
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

    _, model, params, extra_vars = create_model(None, model_name, train_data[0][0].numpy()[None, ...],
                                                num_classes=train_data.n_classes, ckpt_path=ckpt_path)

    full_eval_model(compute_prob_fn(model, params, extra_vars),
                    train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                    log_prefix='s/')


def entrypoint(log_dir=None, **kwargs):
    log_dir, finish_logging = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

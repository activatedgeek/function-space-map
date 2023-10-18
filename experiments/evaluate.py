import wandb

from fspace.nn import get_model
from fspace.utils.logging import entrypoint
from fspace.datasets import get_dataset, get_dataset_attrs, get_loader
from fspace.scripts.evaluate import full_eval_model, compute_prob_fn


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ood_dataset=None, corr_config=None,
         batch_size=512):
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

    train_data, val_data, test_data = get_dataset(dataset, root=data_dir, seed=seed, channels_last=True,
                                                  corr_config=corr_config)
    train_loader = get_loader(train_data, batch_size=batch_size)
    val_loader = get_loader(val_data, batch_size=batch_size) if val_data is not None else None
    test_loader = get_loader(test_data, batch_size=batch_size)

    ood_test_loader = None
    if ood_dataset is not None:
        _, _, ood_test_data = get_dataset(ood_dataset, root=data_dir, seed=seed, channels_last=True,
                                          normalize=get_dataset_attrs(dataset).get('normalize'))
        ood_test_loader = get_loader(ood_test_data, batch_size=batch_size)

    model, params, extra_vars = get_model(model_name, model_dir=ckpt_path,
                                          num_classes=get_dataset_attrs(dataset).get('num_classes'),
                                          inputs=train_data[0][0].numpy()[None, ...])

    full_eval_model(compute_prob_fn(model, params, extra_vars),
                    train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                    log_prefix='s/')


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint(main))

import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax

from optax_swag import swag, sample_swag_diag, sample_swag

from fspace.utils.logging import set_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
from fspace.nn import create_model
from fspace.utils.training import TrainState, train_model
from fspace.scripts.evaluate import \
    cheap_eval_model, full_eval_model, \
    compute_prob_fn, compute_prob_ensemble_fn, \
    compute_mutables_fn


@jax.jit
def train_step_fn(state, b_X, b_Y):
    def loss_fn(params, **extra_vars):
        logits, new_state = state.apply_fn({ 'params': params, **extra_vars }, b_X,
                                            mutable=['batch_stats'], train=True)

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))
        # loss = loss + reg_scale * sum([jnp.vdot(p, p) for p in jax.tree_util.tree_leaves(params)]) / 2

        return loss, new_state

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, **state.extra_vars)

    final_state = state.apply_gradients(grads=grads, **new_state)

    step_metrics = {
        'batch_loss': loss,
    }

    return final_state, step_metrics


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ood_dataset=None,
         train_subset=1., label_noise=0.,
         batch_size=128, num_workers=4,
         optimizer_type='sgd', lr=.1, alpha=0., momentum=.9, reg_scale=0., epochs=0,
         swa_epochs=0, swa_lr=0.05, swag_rank=0, swag_samples=0):
    
    ## SWA updates every epoch.
    swag_rank = min(swa_epochs, swag_rank)
    
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'ood_dataset': ood_dataset,
        'train_subset': train_subset,
        'label_noise': label_noise,
        'batch_size': batch_size,
        'optimizer_type': optimizer_type,
        'lr': lr,
        'alpha': alpha,
        'momentum': momentum,
        'reg_scale': reg_scale,
        'epochs': epochs,
        'swa_epochs': swa_epochs,
        'swa_lr': swa_lr,
        'swag_rank': swag_rank,
        'swag_samples': swag_samples,
    })

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(
        dataset, root=data_dir, seed=seed, train_subset=train_subset, label_noise=label_noise)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    rng, model, init_params, init_vars = create_model(rng, model_name, train_data[0][0].numpy()[None, ...],
                                                      num_classes=train_data.n_classes, ckpt_path=ckpt_path)

    if optimizer_type == 'sgd':
        optimizer = optax.chain(
            optax.add_decayed_weights(reg_scale),
            optax.sgd(learning_rate=optax.cosine_decay_schedule(lr, epochs * len(train_loader), alpha) if epochs else lr, momentum=momentum),
        )
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        **init_vars,
        tx=optimizer)

    train_fn = lambda *args, **kwargs: train_model(*args, train_step_fn, **kwargs)

    for e in tqdm(range(epochs)):
        train_state = train_fn(train_state, train_loader, log_dir=log_dir, epoch=e)

        if (e + 1) % 10 == 0:
            val_metrics = cheap_eval_model(compute_prob_fn(model, train_state.params, train_state.extra_vars), val_loader or test_loader)
            logging.info({ 'epoch': e, **val_metrics }, extra=dict(metrics=True, prefix='val'))
            logging.debug({ 'epoch': e, **val_metrics })

        checkpoints.save_checkpoint(ckpt_dir=log_dir,
                                    target={'params': train_state.params, **train_state.extra_vars},
                                    step=e,
                                    overwrite=True)

    ## Re-init optimizer for SWA.
    if optimizer_type == 'sgd':
        optimizer = optax.chain(
            optax.add_decayed_weights(reg_scale),
            optax.sgd(learning_rate=swa_lr, momentum=momentum),
            swag(len(train_loader), swag_rank),
        )
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=train_state.params,
        **train_state.extra_vars,
        tx=optimizer)

    for e in tqdm(range(swa_epochs)):
        train_state = train_fn(train_state, train_loader, log_dir=log_dir, epoch=e)

        if (e + 1) % 10 == 0:
            _swa_params = jax.tree_map(lambda p: p[jnp.newaxis, ...], train_state.opt_state[-1].mean)
            _extra_vars = jax.tree_map(lambda v: v[jnp.newaxis, ...], train_state.extra_vars)
            _swa_extra_vars = compute_mutables_fn(model, _swa_params)(train_loader, _extra_vars)

            val_metrics = cheap_eval_model(compute_prob_ensemble_fn(model, _swa_params, _swa_extra_vars),
                                           val_loader or test_loader)
            
            logging.info({ 'epoch': epochs + e, **val_metrics }, extra=dict(metrics=True, prefix='val'))
            logging.debug({ 'epoch': epochs + e, **val_metrics })

        checkpoints.save_checkpoint(ckpt_dir=log_dir,
                                    target=train_state.opt_state[-1]._asdict(),
                                    step=epochs + e,
                                    prefix='swa_checkpoint_',
                                    overwrite=True)

    if swa_epochs:
        swa_opt_state = train_state.opt_state[-1]
        
        if swag_samples:
            rng, *samples_rng = jax.random.split(rng, 1 + swag_samples)

            if swag_rank:
                logging.debug(f'Constructing {swag_samples} samples for SWAG evaluation...')

                swag_sample_params = jax.vmap(sample_swag, in_axes=(0, None, None))(
                    jnp.array(samples_rng), swa_opt_state, swag_rank)
            else:
                logging.debug(f'Constructing {swag_samples} samples for SWAG-Diag evaluation...')

                swag_sample_params = jax.vmap(sample_swag_diag, in_axes=(0, None))(
                    jnp.array(samples_rng), swa_opt_state)

            swa_extra_vars = compute_mutables_fn(model, swag_sample_params)(train_loader, 
                                                                            jax.tree_map(lambda v: jnp.repeat(v[jnp.newaxis, ...], swag_samples, 0), train_state.extra_vars))
            c_fn = compute_prob_ensemble_fn(model, swag_sample_params, swa_extra_vars)
        else:
            logging.debug('Constructing SWA evaluation...')

            swa_params = jax.tree_map(lambda p: p[jnp.newaxis, ...], swa_opt_state.mean)
            swa_extra_vars = compute_mutables_fn(model, swa_params)(train_loader, 
                                                                    jax.tree_map(lambda v: v[jnp.newaxis, ...], train_state.extra_vars))
            c_fn = compute_prob_fn(model, swa_params, swa_extra_vars)
    else:
        logging.debug(f'Constructing model evaluation...')

        c_fn = compute_prob_fn(model, train_state.params, train_state.extra_vars)

    ## Full evaluation only at the end of training.
    ood_test_loader = None
    if ood_dataset is not None:
        _, _, ood_test_data = get_dataset(ood_dataset, root=data_dir, seed=seed,
                                          normalize=get_dataset_normalization(dataset))
        ood_test_loader = DataLoader(ood_test_data, batch_size=batch_size, num_workers=num_workers)

    full_eval_model(c_fn,
                    train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                    log_prefix='s/')


def entrypoint(log_dir=None, **kwargs):
    log_dir, finish_logging = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

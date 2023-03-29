import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax
import distrax

from fspace.utils.logging import set_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
from fspace.nn import create_model
from fspace.utils.training import TrainState
from fspace.scripts.evaluate import \
    cheap_eval_model, full_eval_model, \
    compute_prob_fn


@jax.jit
def train_step_fn(state, X, Y, X_ctx, prior_std=1., reg_coef=1., jitter=1e-4):
    def loss_fn(params, **extra_vars):
        X_in = X if X_ctx is None else jnp.concatenate([X, X_ctx], axis=0)

        logits, new_state = state.apply_fn({ 'params': params, **extra_vars }, X_in,
                                            mutable=['batch_stats', 'intermediates'], train=True)

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits[:Y.shape[0]], Y))

        new_state, intermediates = new_state.pop('intermediates')

        phi = jax.lax.stop_gradient(intermediates['features'][0])

        f_h_cov = prior_std**2 * jnp.matmul(phi, phi.T)
        f_h_cov = f_h_cov + prior_std**2 * jnp.ones_like(f_h_cov) + jitter * jnp.eye(f_h_cov.shape[0])
        f_dist = distrax.MultivariateNormalFullCovariance(
            loc=jnp.zeros(f_h_cov.shape[0]), covariance_matrix=f_h_cov)

        reg_loss = - jnp.mean(f_dist.log_prob(logits.T))

        batch_loss = loss + reg_coef * reg_loss

        return batch_loss, { 'state': new_state, 'ce_loss': loss, 'reg_loss': reg_loss }

    (batch_loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, **state.extra_vars)

    new_state = loss_dict.get('state')
    loss_dict.pop('state')

    final_state = state.apply_gradients(grads=grads, **new_state)

    return final_state, { 'batch_loss': batch_loss, **loss_dict }


def train_model(state, loader, step_fn, ctx_loader=None, log_dir=None, epoch=None):
    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.numpy(), Y.numpy()

        X_ctx = None
        if ctx_loader is not None:
            X_ctx, _ = next(ctx_loader.__iter__())
            X_ctx = X_ctx.numpy()

        state, step_metrics = step_fn(state, X, Y, X_ctx)

        if log_dir is not None and i % 100 == 0:
            step_metrics = { k: v.item() for k, v in step_metrics.items() }
            logging.info({ 'epoch': epoch, **step_metrics }, extra=dict(metrics=True, prefix='train'))
            logging.debug({ 'epoch': epoch, **step_metrics })

    return state


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ood_dataset=None, ctx_dataset=None,
         train_subset=1.,
         batch_size=128, num_workers=4,
         f_prior_std=1., jitter=1e-4, f_reg_coef=1.,
         optimizer_type='sgd', lr=.1, alpha=0., momentum=.9, weight_decay=0., epochs=0):
    
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'ctx_dataset': ctx_dataset,
        'ood_dataset': ood_dataset,
        'train_subset': train_subset,
        'batch_size': batch_size,
        'optimizer_type': optimizer_type,
        'lr': lr,
        'alpha': alpha,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'f_prior_std': f_prior_std,
        'f_reg_coef': f_reg_coef,
        'jitter': jitter,
    })

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(dataset, root=data_dir, seed=seed,
                                                  train_subset=train_subset)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    context_loader = None
    if ctx_dataset is not None:
        context_data, _, _ = get_dataset(ctx_dataset, root=data_dir, seed=seed,
                                         normalize=get_dataset_normalization(dataset))
        context_loader = DataLoader(context_data, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True)
            
    rng, model, init_params, init_vars = create_model(rng, model_name, train_data[0][0].numpy()[None, ...],
                                                      num_classes=train_data.n_classes, ckpt_path=ckpt_path)

    if optimizer_type == 'sgd':
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=optax.cosine_decay_schedule(lr, epochs * len(train_loader), alpha) if epochs else lr, momentum=momentum),
        )
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        **init_vars,
        tx=optimizer)

    step_fn = lambda *args: train_step_fn(*args, prior_std=f_prior_std, reg_coef=f_reg_coef, jitter=jitter)
    train_fn = lambda *args, **kwargs: train_model(*args, step_fn, **kwargs)

    for e in tqdm(range(epochs)):
        train_state = train_fn(train_state, train_loader, ctx_loader=context_loader,
                               log_dir=log_dir, epoch=e)

        if (e + 1) % 10 == 0:
            val_metrics = cheap_eval_model(compute_prob_fn(model, train_state.params, train_state.extra_vars), val_loader or test_loader)
            logging.info({ 'epoch': e, **val_metrics }, extra=dict(metrics=True, prefix='val'))
            logging.debug({ 'epoch': e, **val_metrics })

        checkpoints.save_checkpoint(ckpt_dir=log_dir,
                                    target={'params': train_state.params, **train_state.extra_vars},
                                    step=e,
                                    overwrite=True)

    ## Full evaluation only at the end of training.
    ood_test_loader = None
    if ood_dataset is not None:
        _, _, ood_test_data = get_dataset(ood_dataset, root=data_dir, seed=seed,
                                          normalize=get_dataset_normalization(dataset))
        ood_test_loader = DataLoader(ood_test_data, batch_size=batch_size, num_workers=num_workers)

    full_eval_model(compute_prob_fn(model, train_state.params, train_state.extra_vars),
                    train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                    log_prefix='s/')


def entrypoint(log_dir=None, **kwargs):
    log_dir, finish_logging = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

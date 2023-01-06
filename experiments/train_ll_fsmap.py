import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
import optax
import distrax
import timm

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
from fspace.nn import create_model
from fspace.utils.training import TrainState, eval_classifier
from fspace.utils.random import tree_split, sample_tree


@jax.jit
def train_step_fn(rng_tree, state, b_X, b_Y, b_X_ctx, reg_scale, prior_var, jitter=1e-4):
    B = b_X.shape[0]

    def loss_fn(params, **extra_vars):
        b_X_in = b_X if b_X_ctx is None else jnp.concatenate([b_X, b_X_ctx], axis=0)

        b_logits, new_state = state.apply_fn({ 'params': params, **extra_vars }, b_X_in,
                                             mutable=['batch_stats'], train=True)

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(b_logits[:B], b_Y))

        ## FIXME: assumes normal prior.
        prior_params = sample_tree(rng_tree, state.params, 0., 1e-3)
        h_fn = lambda _p, _x: state.apply_fn({ 'params': _p, **extra_vars }, _x, mutable=['batch_stats', 'intermediates'], train=True)[1]['intermediates']['features'][0]
        h_X = jax.lax.stop_gradient(h_fn(prior_params, b_X_in))

        ## FIXME: verify correctness and find a cleaner way.
        H = h_X.shape[-1]
        C = b_logits.shape[-1]
        final_layer_prior_var = jax.tree_map(lambda x: x * 0 + prior_var, jax.lax.stop_gradient(params["Dense_0"]["kernel"]))
        diag_mat = jnp.diagflat(final_layer_prior_var).reshape(H, C, H, C).transpose(1, 3, 0, 2)
        f_cov = jnp.matmul(jnp.matmul(h_X, diag_mat), h_X.T).transpose(2, 0, 3, 1)

        ## FIXME: assumes zero prior mean.
        def f_prior_fn(_cov, _logits):
            dist = distrax.MultivariateNormalFullCovariance(
                loc=jnp.zeros(_cov.shape[0]), covariance_matrix=_cov + jitter * jnp.eye(_cov.shape[0]))
            return dist.log_prob(_logits)
        reg_loss = - jnp.sum(jax.vmap(f_prior_fn)(
            jnp.stack([f_cov[:, c, :, c] for c in range(b_logits.shape[-1])]), b_logits.T))

        total_loss = loss + reg_scale * reg_loss

        return total_loss, new_state

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, **state.extra_vars)

    final_state = state.apply_gradients(grads=grads, **new_state)

    return final_state, loss


def train_model(rng, state, loader, step_fn, ctx_loader=None, log_dir=None, epoch=None):
    ctx_iter = iter(ctx_loader or [[None, None]])

    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.numpy(), Y.numpy()
        try:
            X_ctx, _ = next(ctx_iter)
        except StopIteration:
            ctx_iter = iter(ctx_loader or [[None, None]])
            X_ctx, _ = next(ctx_iter)
        if X_ctx is not None:
            X_ctx = X_ctx.numpy()

        rng, rng_tree = tree_split(rng, state.params)

        state, loss = step_fn(rng_tree, state, X, Y, X_ctx)

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item() }
            logging.info(metrics, extra=dict(metrics=True, prefix='train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ctx_dataset=None, train_subset=1., label_noise=0.,
         batch_size=128, num_workers=4,
         optimizer='sgd', lr=.1, momentum=.9, alpha=0., reg_scale=0., prior_var=1.,
         epochs=0):
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'ctx_dataset': ctx_dataset,
        'train_subset': train_subset,
        'label_noise': label_noise,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'lr': lr,
        'momentum': momentum,
        'alpha': alpha,
        'reg_scale': reg_scale,
        'prior_var': prior_var,
        'epochs': epochs,
    })

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(
        dataset, root=data_dir, seed=seed, train_subset=train_subset, label_noise=label_noise)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    ctx_loader = None
    if ctx_dataset is not None:
        ctx_data = get_dataset(ctx_dataset, root=data_dir, is_ctx=True, batch_size=batch_size, seed=seed,
                               normalize=get_dataset_normalization(dataset))
        ctx_loader = DataLoader(ctx_data, batch_size=batch_size, num_workers=num_workers,
                                shuffle=not isinstance(ctx_data, timm.data.IterableImageDataset))
        logging.debug(f'Using {ctx_dataset} for context samples.')

    model = create_model(model_name, num_classes=train_data.n_classes)
    if ckpt_path is not None:
        init_vars = freeze(checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None))
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_vars = model.init(model_init_rng, train_data[0][0].numpy()[None, ...])
    other_vars, params = init_vars.pop('params')

    if optimizer == 'adamw':
        optimizer = optax.adamw(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = optax.sgd(learning_rate=optax.cosine_decay_schedule(lr, epochs * len(train_loader), alpha), momentum=momentum)
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        **other_vars,
        tx=optimizer)

    step_fn = lambda *args: train_step_fn(*args, reg_scale, prior_var)
    train_fn = lambda *args, **kwargs: train_model(rng, *args, train_loader, step_fn,
                                                   ctx_loader=ctx_loader, log_dir=log_dir, **kwargs)

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        train_state = train_fn(train_state, epoch=e)

        val_metrics = eval_classifier(train_state, val_loader if val_loader.dataset is not None else test_loader)
        logging.info({ 'epoch': e, **val_metrics }, extra=dict(metrics=True, prefix='val'))

        if val_metrics['acc'] > best_acc_so_far:
            best_acc_so_far = val_metrics['acc']

            train_metrics = eval_classifier(train_state, train_loader)
            logging.info({ 'epoch': e, **train_metrics }, extra=dict(metrics=True, prefix='train'))

            test_metrics = eval_classifier(train_state, test_loader)
            logging.info({ 'epoch': e, **test_metrics }, extra=dict(metrics=True, prefix='test'))

            wandb.run.summary['val/best_epoch'] = e
            wandb.run.summary['train/best_acc'] = train_metrics['acc']
            wandb.run.summary['val/best_acc'] = val_metrics['acc']
            wandb.run.summary['test/best_acc'] = test_metrics['acc']

            logging.info(f"Epoch {e}: {train_metrics['acc']:.4f} (Train) / {val_metrics['acc']:.4f} (Val) / {test_metrics['acc']:.4f} (Test)")

            checkpoints.save_checkpoint(ckpt_dir=log_dir,
                                        target={'params': train_state.params, **train_state.extra_vars},
                                        step=e,
                                        overwrite=True)


def entrypoint(log_dir=None, **kwargs):
    log_dir = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax
import timm

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset, get_dataset_normalization
from fspace.nn import create_model
from fspace.utils.training import TrainState, eval_classifier
from fspace.scripts.evaluate import full_eval_model


@jax.jit
def train_step_fn(_, state, b_X, b_Y, b_X_ctx, reg_scale):
    B = b_X.shape[0]

    def loss_fn(params, **extra_vars):
        b_X_in = b_X if b_X_ctx is None else jnp.concatenate([b_X, b_X_ctx], axis=0)

        b_logits, new_state = state.apply_fn({ 'params': params, **extra_vars }, b_X_in,
                                             mutable=['batch_stats'], train=True)

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(b_logits[:B], b_Y))

        reg_loss = jnp.sum(jnp.square(b_logits))

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

        state, loss = step_fn(rng, state, X, Y, X_ctx)

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item() }
            logging.info(metrics, extra=dict(metrics=True, prefix='train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, ctx_dataset=None, ood_dataset=None,
         train_subset=1., label_noise=0.,
         batch_size=128, num_workers=4,
         optimizer='sgd', lr=.1, momentum=.9, alpha=0., reg_scale=0.,
         epochs=0):
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'ctx_dataset': ctx_dataset,
        'ood_dataset': ood_dataset,
        'train_subset': train_subset,
        'label_noise': label_noise,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'lr': lr,
        'momentum': momentum,
        'alpha': alpha,
        'reg_scale': reg_scale,
        'epochs': epochs,
    })

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(
        dataset, root=data_dir, seed=seed, train_subset=train_subset, label_noise=label_noise)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    ctx_loader = None
    if ctx_dataset is not None:
        ctx_data = get_dataset(ctx_dataset, root=data_dir, is_ctx=True, batch_size=batch_size, seed=seed,
                               normalize=get_dataset_normalization(dataset))
        ctx_loader = DataLoader(ctx_data, batch_size=batch_size, num_workers=num_workers,
                                shuffle=not isinstance(ctx_data, timm.data.IterableImageDataset))
        logging.debug(f'Using {ctx_dataset} for context samples.')

    rng, model, init_params, init_vars = create_model(rng, model_name, train_data[0][0].numpy()[None, ...],
                                                      num_classes=train_data.n_classes, ckpt_path=ckpt_path)

    if optimizer == 'adamw':
        optimizer = optax.adamw(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = optax.sgd(learning_rate=optax.cosine_decay_schedule(lr, epochs * len(train_loader), alpha), momentum=momentum)
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        **init_vars,
        tx=optimizer)

    step_fn = lambda *args: train_step_fn(*args, reg_scale)
    train_fn = lambda *args, **kwargs: train_model(rng, *args, train_loader, step_fn,
                                                   ctx_loader=ctx_loader, log_dir=log_dir, **kwargs)

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        train_state = train_fn(train_state, epoch=e)

        val_metrics = eval_classifier(train_state, val_loader or test_loader)
        logging.info({ 'epoch': e, **val_metrics }, extra=dict(metrics=True, prefix='val'))

        if val_metrics['acc'] > best_acc_so_far:
            best_acc_so_far = val_metrics['acc']

            logging.info(f"Epoch {e}: {best_acc_so_far:.4f} (Val)")

            checkpoints.save_checkpoint(ckpt_dir=log_dir,
                                        target={'params': train_state.params, **train_state.extra_vars},
                                        step=e,
                                        prefix='best_checkpoint_',
                                        overwrite=True)

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

    logging.info(f'Evaluating latest checkpoint...')
    full_eval_model(compute_p_fn(model_name, train_data.n_classes, log_dir, ckpt_prefix='checkpoint_'),
                    train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                    log_prefix='s/')

    try:
        logging.info(f'Evaluating best (validation) checkpoint...')
        full_eval_model(compute_p_fn(model_name, train_data.n_classes, log_dir, ckpt_prefix='best_checkpoint_'),
                        train_loader, test_loader, val_loader=val_loader, ood_loader=ood_test_loader,
                        log_prefix='s/best/')
    except TypeError:
        logging.warning('Skipping best checkpoint evaluation.')


def entrypoint(log_dir=None, **kwargs):
    log_dir, finish_logging = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

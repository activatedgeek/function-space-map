import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset
from fspace.nn import MResNet18
from fspace.utils.training import TrainState, train_model, eval_model


@jax.jit
def train_step_fn(state, b_X, b_Y, func_decay):
    def loss_fn(params, batch_stats):
        logits, new_state = state.apply_fn({ 'params': params, 'batch_stats': batch_stats }, b_X,
                                            mutable=['batch_stats'], train=True)

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))
        loss = loss + func_decay * jnp.vdot(logits, logits) / 2

        return loss, new_state

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params,state.batch_stats)

    final_state = state.apply_gradients(grads=grads, batch_stats=new_state['batch_stats'])

    return final_state, loss


@jax.jit
def eval_step_fn(state, b_X, b_Y):
    logits = state.apply_fn({ 'params': state.params, 'batch_stats': state.batch_stats}, b_X,
                            mutable=False, train=False)

    nll = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))

    return logits, nll


def main(seed=42, log_dir=None, data_dir=None,
         ckpt_path=None,
         dataset=None, batch_size=128, num_workers=4,
         optimizer='sgd', lr=.1, momentum=.9, func_decay=0.,
         epochs=0):

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(dataset, root=data_dir, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    model = MResNet18(num_classes=train_data.n_classes)
    if ckpt_path is not None:
        init_variables = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_variables = model.init(model_init_rng, train_data[0][0].numpy()[None, ...])

    if optimizer == 'adamw':
        optimizer = optax.adamw(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = optax.sgd(learning_rate=optax.cosine_decay_schedule(lr, epochs * len(train_loader), 1e-4), momentum=momentum)
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=init_variables['params'],
        batch_stats=init_variables['batch_stats'],
        tx=optimizer)

    step_fn = lambda *args: train_step_fn(*args, func_decay)
    train_fn = lambda *args, **kwargs: train_model(*args, step_fn, **kwargs)
    eval_fn = lambda *args: eval_model(*args, eval_step_fn)

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        train_state = train_fn(train_state, train_loader, log_dir=log_dir, epoch=e)
        
        val_metrics = eval_fn(train_state, val_loader)
        logging.info({ 'epoch': e, **val_metrics }, extra=dict(metrics=True, prefix='sgd/val'))

        if val_metrics['acc'] > best_acc_so_far:
            best_acc_so_far = val_metrics['acc']

            train_metrics = eval_fn(train_state, train_loader)
            logging.info({ 'epoch': e, **train_metrics }, extra=dict(metrics=True, prefix='sgd/train'))

            test_metrics = eval_fn(train_state, test_loader)
            logging.info({ 'epoch': e, **test_metrics }, extra=dict(metrics=True, prefix='sgd/test'))

            wandb.run.summary['val/best_epoch'] = e
            wandb.run.summary['train/best_acc'] = train_metrics['acc']
            wandb.run.summary['val/best_acc'] = val_metrics['acc']
            wandb.run.summary['test/best_acc'] = test_metrics['acc']

            logging.info(f"Epoch {e}: {train_metrics['acc']:.4f} (Train) / {val_metrics['acc']:.4f} (Val) / {test_metrics['acc']:.4f} (Test)")

            checkpoints.save_checkpoint(ckpt_dir=log_dir,
                                        target={'params': train_state.params,
                                                'batch_stats': train_state.batch_stats},
                                        step=e,
                                        overwrite=True)


def entrypoint(log_dir=None, **kwargs):
    log_dir = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

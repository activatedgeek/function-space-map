import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset
from fspace.nn import create_model
from fspace.utils.training import TrainState, eval_model
from fspace.utils.random import tree_split


@jax.jit
def train_step_fn(rng_trees, state, b_X, b_Y, func_decay, noise_std=1e-4):
    def sample_delta_fn(rng_tree, logits):
        ## @NOTE: Assumes zero mean param.
        def sample_param(key, param):
            return noise_std * jax.random.normal(key, param.shape, param.dtype)

        sample_params = jax.tree_util.tree_map(sample_param, rng_tree, state.params)
        sample_logits = state.apply_fn({ 'params': sample_params, **state.extra_vars}, b_X,
                                        mutable=False, train=False)
        sample_delta = logits - jax.lax.stop_gradient(sample_logits)
        return sample_delta

    def loss_fn(params, **extra_vars):
        logits, new_state = state.apply_fn({ 'params': params, **extra_vars }, b_X,
                                            mutable=['batch_stats'], train=True)

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))

        if len(rng_trees):
            reg_loss = []
            for rng_tree in rng_trees:
                sample_delta = sample_delta_fn(rng_tree, logits)
                reg_loss.append(jnp.vdot(sample_delta, sample_delta))
            reg_loss = - jax.nn.logsumexp(- func_decay * jnp.array(reg_loss) / 2)
        else:
            reg_loss = func_decay * jnp.vdot(logits, logits) / 2

        loss = loss + reg_loss

        return loss, new_state

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, **state.extra_vars)

    final_state = state.apply_gradients(grads=grads, **new_state)

    return final_state, loss


@jax.jit
def eval_step_fn(state, b_X, b_Y):
    logits = state.apply_fn({ 'params': state.params, **state.extra_vars}, b_X,
                            mutable=False, train=False)

    nll = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))

    return logits, nll


def train_model(rng, state, loader, step_fn, n_samples, log_dir=None, epoch=None):
    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        rng_trees = []
        for _ in range(n_samples):
            rng, _tree = tree_split(rng, state.params)
            rng_trees.append(_tree)

        X, Y = X.numpy(), Y.numpy()

        state, loss = step_fn(rng_trees, state, X, Y)

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item() }
            logging.info(metrics, extra=dict(metrics=True, prefix='sgd/train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, train_subset=1., label_noise=0.,
         batch_size=128, num_workers=4,
         optimizer='sgd', lr=.1, momentum=.9, func_decay=0., n_samples=0, noise_std=1e-4,
         epochs=0):
    wandb.config.update({
        'log_dir': log_dir,
        'seed': seed,
        'model_name': model_name,
        'ckpt_path': ckpt_path,
        'dataset': dataset,
        'train_subset': train_subset,
        'label_noise': label_noise,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'lr': lr,
        'momentum': momentum,
        'func_decay': func_decay,
        'n_samples': n_samples,
        'noise_std': noise_std,
        'epochs': epochs,
    })

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(
        dataset, root=data_dir, seed=seed, train_subset=train_subset, label_noise=label_noise)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    model = create_model(model_name, num_classes=train_data.n_classes)
    if ckpt_path is not None:
        init_vars = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_vars = model.init(model_init_rng, train_data[0][0].numpy()[None, ...])
    other_vars, params = init_vars.pop('params')

    if optimizer == 'adamw':
        optimizer = optax.adamw(learning_rate=lr)
    elif optimizer == 'sgd':
        optimizer = optax.sgd(learning_rate=optax.cosine_decay_schedule(lr, epochs * len(train_loader), 1e-3), momentum=momentum)
    # elif optimizer == 'lro':
    #     from learned_optimization.research.general_lopt import prefab
    #     optimizer = prefab.optax_lopt(epochs * len(train_loader))
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        **other_vars,
        tx=optimizer)

    step_fn = lambda *args: train_step_fn(*args, func_decay, noise_std=noise_std)
    train_fn = lambda *args, **kwargs: train_model(rng, *args, step_fn, n_samples, **kwargs)
    eval_fn = lambda *args: eval_model(*args, eval_step_fn)

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        train_state = train_fn(train_state, train_loader, log_dir=log_dir, epoch=e)
        
        val_metrics = eval_fn(train_state, val_loader if val_loader.dataset is not None else test_loader)
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

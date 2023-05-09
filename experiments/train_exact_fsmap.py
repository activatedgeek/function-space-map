import logging
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax

from fspace.utils.logging import set_logging, wandb
from fspace.datasets import get_dataset, get_dataset_attrs, get_loader
from fspace.nn import create_model
from fspace.utils.training import TrainState

def jacobian_sigular_values(jac_fn, p, x, **extra_vars):
    j = jac_fn(p, x, **extra_vars)
    j = jax.tree_util.tree_map(lambda x: jnp.einsum('b...->...b', x), j)
    # flatten j
    J, _ = jax.flatten_util.ravel_pytree(j)
    num_params = jax.flatten_util.ravel_pytree(p)[0].shape[0]
    J = J.reshape(num_params, -1).T # (N, P)
    print('J:', J.shape)

    # sigular values of J
    S = jnp.linalg.svd(J, compute_uv=False)
    print('S:', S.shape)
    return S

def log_det_H(jac_fn, p, x, jitter, **extra_vars):
    # log det J^T J = sum log s^2 (careful, check this for more general cases when J is not injective)
    s = jacobian_sigular_values(jac_fn, p, x, **extra_vars)
    s = s + jitter
    logdet_svd = 2 * jnp.sum(jnp.log(s))
    return logdet_svd

def log_frobenius_J(jac_fn, p, x, jitter, **extra_vars):
    # s(J) <= ||J||_f, log s(J) <= log ||J||_f
    j = jac_fn(p, x, **extra_vars)
    j = jax.tree_util.tree_map(lambda x: jnp.einsum('b...->...b', x), j)
    # flatten j
    J, _ = jax.flatten_util.ravel_pytree(j)
    num_params = jax.flatten_util.ravel_pytree(p)[0].shape[0]
    J = J.reshape(num_params, -1).T # (N, P)
    print('J:', J.shape)
    f_norm = jnp.linalg.norm(J, ord='fro')
    # 2 * jnp.sum(jnp.log(s)) <= 2 * min(n * c, p) *jnp.log(f_norm + jitter)
    return 2 * jnp.log(f_norm + jitter) #* min(J.shape[0], J.shape[1])

def train_step_fn(state, b_X, b_Y, b_X_eval, log_det_fn, n_train):
    def loss_fn(params, **extra_vars):
        logits, new_state = state.apply_fn({ 'params': params, **extra_vars }, b_X,
                                            mutable=['batch_stats'], train=True)
        print('out shape:', logits.shape)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))
        fs_loss = 1 / 2 * log_det_fn(params, b_X_eval, **extra_vars) / n_train
        loss = (loss + fs_loss) 

        return loss, (fs_loss, new_state)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, **state.extra_vars)
    fs_loss, new_state = aux

    final_state = state.apply_gradients(grads=grads, **new_state)

    return final_state, loss, fs_loss

def train_model(state, loader, step_fn, log_dir=None, epoch=None):
    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.numpy(), Y.numpy()
        X_eval = X
        state, loss, fs_loss = step_fn(state, X, Y, X_eval)
        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item(), 'fs_loss': fs_loss.item(), 'ce_loss': loss.item() - fs_loss.item()}
            logging.info(metrics, extra=dict(metrics=True, prefix='sgd/train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


def main(seed=42, log_dir=None, data_dir=None,
         model_name=None, ckpt_path=None,
         dataset=None, train_subset=1., label_noise=0.,
         batch_size=128,
         optimizer='sgd', lr=.1, momentum=.9, weight_decay=0., jitter=1e-5,
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
        'weight_decay': weight_decay,
        'epochs': epochs,
    })
    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(
        dataset, root=data_dir, seed=seed, train_subset=train_subset, label_noise=label_noise)
    train_loader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = get_loader(val_data, batch_size=batch_size)
    test_loader = get_loader(test_data, batch_size=batch_size)

    model = create_model(model_name, num_classes=get_dataset_attrs(dataset).get('num_classes'))
    if ckpt_path is not None:
        init_vars = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_vars = model.init(model_init_rng, train_data[0][0].numpy()[None, ...])
    params = init_vars.pop('params')
    if len(params) == 2:
        other_vars, params = params
    else:
        other_vars = {}

    if optimizer == 'adamw':
        optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=optax.cosine_decay_schedule(lr, epochs * len(train_loader), 1e-3), momentum=momentum),
        )
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        **other_vars,
        tx=optimizer)
    
    jac_fn = jax.jacrev(lambda p, x, **extra_vars: model.apply({'params': p, **extra_vars}, x, train=False).reshape(-1))
    step_fn = jax.jit(lambda *args: train_step_fn(*args, log_det_fn=lambda p, x, **extra: log_det_H(jac_fn, p, x, jitter, **extra), n_train=len(train_data)))
    train_fn = lambda *args, **kwargs: train_model(*args, step_fn, **kwargs)

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        train_state = train_fn(train_state, train_loader, log_dir=log_dir, epoch=e)
        
        val_metrics = eval_classifier(train_state, val_loader if val_loader.dataset is not None else test_loader)
        logging.info({ 'epoch': e, **val_metrics }, extra=dict(metrics=True, prefix='sgd/val'))

        if val_metrics['acc'] > best_acc_so_far:
            best_acc_so_far = val_metrics['acc']

            train_metrics = eval_classifier(train_state, train_loader)
            logging.info({ 'epoch': e, **train_metrics }, extra=dict(metrics=True, prefix='sgd/train'))

            test_metrics = eval_classifier(train_state, test_loader)
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
    log_dir, finish_logging = set_logging(log_dir=log_dir)

    main(**kwargs, log_dir=log_dir)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

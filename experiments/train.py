import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import flaxmodels as fm

from fspace.utils.logging import set_logging, finish_logging, wandb
from fspace.datasets import get_dataset


## Override to have batch statistics.
class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict


@jax.jit
def train_step_fn(state, b_X, b_Y):
    def loss_fn(params, batch_stats):
        logits, new_state = state.apply_fn({ 'params': params, 'batch_stats': batch_stats }, b_X,
                                            train=True, mutable=['batch_stats'])

        
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))
        # loss = loss + weight_decay * sum([jnp.vdot(p, p) for p in jax.tree_util.tree_leaves(params)]) / 2

        return loss, new_state

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params,state.batch_stats)

    final_state = state.apply_gradients(grads=grads, batch_stats=new_state['batch_stats'])

    return final_state, loss


def train_model(state, loader, step_fn=train_step_fn, log_dir=None, epoch=None):
    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.numpy(), Y.numpy()

        state, loss = step_fn(state, X, Y)

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.item() }
            logging.info(metrics, extra=dict(metrics=True, prefix='sgd/train'))
            logging.debug(f'Epoch {epoch}: {loss.item():.4f}')

    return state


@jax.jit
def eval_step_fn(state, b_X, b_Y):
    logits = state.apply_fn({ 'params': state.params, 'batch_stats': state.batch_stats}, b_X,
                            train=False, mutable=False)

    nll = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, b_Y))

    return logits, nll


def eval_model(state, loader, step_fn=eval_step_fn):
    N = len(loader.dataset)
    N_acc = 0
    nll = 0.

    for X, Y in tqdm(loader, leave=False):
        X, Y = X.numpy(), Y.numpy()

        _logits, _nll = step_fn(state, X, Y)
        pred_Y = jnp.argmax(_logits, axis=-1)
        N_acc += jnp.sum(pred_Y == Y)
        nll += _nll

    return { 'acc': N_acc / N, 'nll': nll, 'avg_nll': nll / N }


def main(seed=42, log_dir=None, data_dir=None,
         ckpt_path=None,
         dataset=None, batch_size=128, num_workers=4,
         optimizer='sgd', lr=.1, momentum=.9, weight_decay=0.,
         epochs=0):

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(dataset, root=data_dir, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    model = fm.ResNet18(
        output='logits', num_classes=train_data.n_classes,
        pretrained=None, normalize=False, kernel_init=nn.initializers.kaiming_normal())
    if ckpt_path is not None:
        init_variables = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_variables = model.init(model_init_rng, train_data[0][0].numpy()[None, ...], train=False)

    if optimizer == 'adamw':
        optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=lr, momentum=momentum),
        )
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=init_variables['params'],
        batch_stats=init_variables['batch_stats'],
        tx=optimizer)

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        train_state = train_model(train_state, train_loader, log_dir=log_dir, epoch=e)
        
        val_metrics = eval_model(train_state, val_loader)
        logging.info(val_metrics, extra=dict(metrics=True, prefix='sgd/val'))

        if val_metrics['acc'] > best_acc_so_far:
            best_acc_so_far = val_metrics['acc']

            train_metrics = eval_model(train_state, train_loader)
            logging.info(train_metrics, extra=dict(metrics=True, prefix='sgd/train'))

            test_metrics = eval_model(train_state, test_loader)
            logging.info(test_metrics, extra=dict(metrics=True, prefix='sgd/test'))

            wandb.run.summary['val/best_epoch'] = e
            wandb.run.summary['train/best_acc'] = train_metrics['acc']
            wandb.run.summary['val/best_acc'] = val_metrics['acc']
            wandb.run.summary['test/best_acc'] = test_metrics['acc']

            logging.info(f"Epoch {e}: {best_acc_so_far:.4f} (Val) / {test_metrics['acc']:.4f} (Test)")

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

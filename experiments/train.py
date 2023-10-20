import logging
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb

from fspace.utils.logging import entrypoint
from fspace.datasets import get_dataset, get_dataset_attrs, get_loader
from fspace.nn import get_model
from fspace.utils.training import TrainState
from fspace.scripts.evaluate import cheap_eval_model, full_eval_model, compute_prob_fn


@jax.jit
def train_step_fn(rng, state, X, Y, X_ctx, laplace_std=1e-2, reg_scale=1e-4):
    X_in = X if X_ctx is None else jnp.concatenate([X, X_ctx], axis=0)

    def tree_random_split(key, ref_tree):
        treedef = jax.tree_util.tree_structure(ref_tree)
        key, *key_list = jax.random.split(key, 1 + treedef.num_leaves)
        return key, jax.tree_util.tree_unflatten(treedef, key_list)

    def tree_random_normal(key, ref_tree, std=1.0):
        _, key_tree = tree_random_split(key, ref_tree)
        return jax.tree_util.tree_map(
            lambda k, v: std * jax.random.normal(k, v.shape, v.dtype),
            key_tree,
            ref_tree,
        )

    def loss_fn(params, extra_vars):
        logits, mutables = state.apply_fn(
            {"params": params, **extra_vars}, X_in, mutable=["batch_stats"], train=True
        )

        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits[: Y.shape[0]], Y)
        )

        perturbed_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, tree_random_normal(rng, params, std=laplace_std)
        )

        perturbed_logits, _ = state.apply_fn(
            {"params": perturbed_params, **extra_vars},
            X_in,
            mutable=["batch_stats"],
            train=True,
        )

        reg_loss = (
            jnp.mean(jnp.sum((perturbed_logits - logits) ** 2, axis=-1))
            / laplace_std**2
        )

        batch_loss = loss + reg_scale * reg_loss

        return batch_loss, {
            "mutables": mutables,
            "batch_loss": batch_loss,
            "ce_loss": loss,
            "reg_loss": reg_loss,
        }

    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.extra_vars
    )
    mutables = aux.pop("mutables")

    final_state = state.apply_gradients(grads=grads, **mutables)

    return final_state, aux


def train_model(rng, state, loader, step_fn, ctx_loader=None, log_dir=None, epoch=None):
    ctx_iter = ctx_loader.__iter__() if ctx_loader is not None else iter([[None, None]])

    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        rng, train_rng = jax.random.split(rng)

        X, Y = X.numpy(), Y.numpy()

        try:
            X_ctx, _ = next(ctx_iter)
        except StopIteration:
            ctx_iter = (
                ctx_loader.__iter__()
                if ctx_loader is not None
                else iter([[None, None]])
            )
            X_ctx, _ = next(ctx_iter)
        if X_ctx is not None:
            X_ctx = X_ctx.numpy()

        state, step_metrics = step_fn(train_rng, state, X, Y, X_ctx)

        if log_dir is not None and i % 100 == 0:
            step_metrics = {k: v.item() for k, v in step_metrics.items()}
            logging.info(
                {"epoch": epoch, **step_metrics},
                extra=dict(metrics=True, prefix="train"),
            )
            logging.debug({"epoch": epoch, **step_metrics})

    return rng, state


def main(
    seed=42,
    log_dir=None,
    save_steps=10,
    data_dir=None,
    model_name=None,
    model_dir=None,
    dataset=None,
    ood_dataset=None,
    ctx_dataset=None,
    augment=True,
    label_noise=0.0,
    batch_size=128,
    context_size=128,
    laplace_std=1e-3,
    reg_scale=0.0,
    optimizer_type="sgd",
    lr=0.1,
    alpha=0.0,
    momentum=0.9,
    weight_decay=0.0,
    epochs=0,
):
    wandb.config.update(
        {
            "log_dir": log_dir,
            "seed": seed,
            "model_name": model_name,
            "model_dir": model_dir,
            "dataset": dataset,
            "ctx_dataset": ctx_dataset,
            "ood_dataset": ood_dataset,
            "augment": bool(augment),
            "label_noise": label_noise,
            "batch_size": batch_size,
            "context_size": context_size,
            "optimizer_type": optimizer_type,
            "lr": lr,
            "alpha": alpha,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "laplace_std": laplace_std,
            "reg_scale": reg_scale,
        }
    )

    rng = jax.random.PRNGKey(seed)

    train_data, val_data, test_data = get_dataset(
        dataset,
        root=data_dir,
        seed=seed,
        channels_last=True,
        augment=bool(augment),
        label_noise=label_noise,
    )
    train_loader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = (
        get_loader(val_data, batch_size=batch_size) if val_data is not None else None
    )
    test_loader = get_loader(test_data, batch_size=batch_size)

    context_loader = None
    if ctx_dataset is not None:
        context_data, _, _ = get_dataset(
            ctx_dataset,
            root=data_dir,
            seed=seed,
            channels_last=True,
            normalize=get_dataset_attrs(dataset).get("normalize"),
            ref_tensor=train_data[0][0],
        )

        context_loader = get_loader(context_data, batch_size=context_size, shuffle=True)

    rng, model_rng = jax.random.split(rng)
    model, init_params, init_vars = get_model(
        model_name,
        model_dir=model_dir,
        num_classes=get_dataset_attrs(dataset).get("num_classes"),
        inputs=train_data[0][0].numpy()[None, ...],
        init_rng=model_rng,
    )

    if optimizer_type == "sgd":
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(
                learning_rate=optax.cosine_decay_schedule(
                    lr, epochs * len(train_loader), alpha
                )
                if epochs
                else lr,
                momentum=momentum,
            ),
            optax.clip_by_global_norm(1.0),
        )
    elif optimizer_type == "adamw":
        optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError

    train_state = TrainState.create(
        apply_fn=model.apply, params=init_params, **init_vars, tx=optimizer
    )

    step_fn = lambda *args: train_step_fn(
        *args, laplace_std=laplace_std, reg_scale=reg_scale
    )
    train_fn = lambda *args, **kwargs: train_model(*args, step_fn, **kwargs)

    checkpointer = ocp.CheckpointManager(
        log_dir,
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(step_prefix="checkpoint"),
    )

    for e in tqdm(range(epochs)):
        rng, train_state = train_fn(
            rng,
            train_state,
            train_loader,
            ctx_loader=context_loader,
            log_dir=log_dir,
            epoch=e,
        )

        if (e + 1) % 10 == 0:
            val_metrics = cheap_eval_model(
                compute_prob_fn(model, train_state.params, train_state.extra_vars),
                val_loader or test_loader,
            )
            logging.info(
                {"epoch": e, **val_metrics}, extra=dict(metrics=True, prefix="val")
            )
            logging.debug({"epoch": e, **val_metrics})

        if (e + 1) % save_steps == 0:
            checkpointer.save(e + 1, train_state.state_dict)

    ## Full evaluation only at the end of training.
    ood_test_loader = None
    if ood_dataset is not None:
        _, _, ood_test_data = get_dataset(
            ood_dataset,
            root=data_dir,
            seed=seed,
            channels_last=True,
            normalize=get_dataset_attrs(dataset).get("normalize"),
        )
        ood_test_loader = get_loader(ood_test_data, batch_size=batch_size)

    full_eval_model(
        compute_prob_fn(model, train_state.params, train_state.extra_vars),
        train_loader,
        test_loader,
        val_loader=val_loader,
        ood_loader=ood_test_loader,
        log_prefix="s/",
    )


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))

import os
import logging
from functools import wraps
import json
import torch
import flax
from flax.training import checkpoints


__all__ = [
    "register_model",
    "get_model",
    "get_model_attrs",
]


MODEL_CONFIG_NAME = "model_config.json"
MODEL_NAME = "model.pt"


__func_map = dict()
__attr_map = dict()


def register_model(function=None, attrs=None, **d_kwargs):
    def _decorator(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            all_kwargs = {**d_kwargs, **kwargs}
            return f(*args, **all_kwargs)

        assert (
            _wrapper.__name__ not in __func_map
        ), f'Duplicate registration for "{_wrapper.__name__}"'

        __func_map[_wrapper.__name__] = _wrapper
        __attr_map[_wrapper.__name__] = attrs
        return _wrapper

    if function:
        return _decorator(function)
    return _decorator


def get_model_fn(name):
    if name not in __func_map:
        raise ValueError(f'Model "{name}" not found.')

    return __func_map[name]


def get_model_attrs(name):
    if name not in __attr_map:
        raise ValueError(f'Model "{name}" attributes not found.')

    return __attr_map[name]


def list_models():
    return list(__func_map.keys())


def get_model(
    model_name=None, model_dir=None, inputs=None, init_rng=None, **model_config
):
    model_config = {"model_name": model_name, **model_config}

    if model_dir is not None and os.path.isfile(f"{model_dir}/{MODEL_CONFIG_NAME}"):
        with open(f"{model_dir}/{MODEL_CONFIG_NAME}") as f:
            model_config = {**model_config, **json.load(f)}

        logging.info(
            f"Loaded model configuration from '{model_dir}/{MODEL_CONFIG_NAME}'."
        )

    model_fn = get_model_fn(model_config.pop("model_name"))

    model = model_fn(**model_config)

    if isinstance(model, torch.nn.Module):
        if model_dir is not None and os.path.isfile(f"{model_dir}/{MODEL_NAME}"):
            model.load_state_dict(torch.load(f"{model_dir}/{MODEL_NAME}"))

            logging.info(f"Loaded model state from '{model_dir}/{MODEL_NAME}'.")

        logging.info(f'Loaded "{model_name}".')

        return model
    elif isinstance(model, flax.linen.Module):
        if model_dir is not None:
            init_vars = flax.core.frozen_dict.freeze(
                checkpoints.restore_checkpoint(
                    ckpt_dir=model_dir, target=None, prefix="checkpoint_"
                )
            )

            logging.info(f'Loaded checkpoint from "{model_dir}".')
        else:
            init_vars = model.init(init_rng, inputs)

        other_vars, params = init_vars.pop("params")

        logging.info(f'Loaded "{model_name}".')

        return model, params, other_vars
    else:
        raise NotImplementedError

import logging
import jax
from flax.training import checkpoints
from flax.core.frozen_dict import freeze

from .mlp import MLP200
from .cnn import SmallCNN, TinyCNN
from .third_party.resnet import ResNet9, ResNet18, ResNet50


__MODEL_CFG = {
    'mlp200': {
        'model_cls': MLP200,
    },
    'resnet9': {
        'model_cls': ResNet9,
    },
    'resnet18': {
        'model_cls': ResNet18,
    },
    'resnet50': {
        'model_cls': ResNet50,
    },
    'smallcnn': {
        'model_cls': SmallCNN,
    },
    'tinycnn': {
        'model_cls': TinyCNN,
    }
}


def create_model(rng, model_name, sample_input, num_classes=10,
                 ckpt_path=None, ckpt_prefix='checkpoint_', **kwargs):
    assert model_name in __MODEL_CFG, f'Model "{model_name}" not supported'

    model = __MODEL_CFG.get(model_name).get('model_cls')(num_classes=num_classes, **kwargs)

    if ckpt_path is not None:
        init_vars = freeze(checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None, prefix=ckpt_prefix))
        logging.info(f'Loaded checkpoint from "{ckpt_path}".')
    else:
        rng, model_init_rng = jax.random.split(rng)
        init_vars = model.init(model_init_rng, sample_input)
    other_vars, params = init_vars.pop('params')

    ## FIXME: tmp. fix for compatibility
    try:
        other_vars, _ = other_vars.pop('params_logvar')
        logging.warning('Removed "params_logvar".')
    except KeyError:
        pass

    return rng, model, params, other_vars

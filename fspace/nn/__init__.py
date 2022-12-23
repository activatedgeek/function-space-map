from .cnn import SmallCNN, TinyCNN
from .third_party.resnet import ResNet9, ResNet18


__MODEL_CFG = {
    'resnet9': {
        'model_cls': ResNet9,
    },
    'resnet18': {
        'model_cls': ResNet18,
    },
    'smallcnn': {
        'model_cls': SmallCNN,
    },
    'tinycnn': {
        'model_cls': TinyCNN,
    }
}


def create_model(model_name, num_classes=10, **kwargs):
    assert model_name in __MODEL_CFG, f'Model "{model_name}" not supported'

    model_cls = __MODEL_CFG.get(model_name).get('model_cls')

    return model_cls(num_classes=num_classes, **kwargs)

from .cnn import SmallCNN
from .resnet import MResNet18, ResNet18


__MODEL_CFG = {
    'resnet18': {
        'model_cls': ResNet18,
    },
    'mresnet18': {
        'model_cls': MResNet18,
    },
    'smallcnn': {
        'model_cls': SmallCNN,
    }
}

def create_model(model_name, num_classes=10, **kwargs):
    assert model_name in __MODEL_CFG, f'Model "{model_name}" not supported'

    model_cls = __MODEL_CFG.get(model_name).get('model_cls')

    return model_cls(num_classes=num_classes, **kwargs)

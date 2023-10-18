from .registry import get_model

__all__ = [
    "get_model",
]

def __setup():
    from importlib import import_module

    for n in [
        "cnn",
        "mlp",
        "third_party.resnet",
        "third_party.resnet_pretrained",
    ]:
        import_module(f".{n}", __name__)


__setup()
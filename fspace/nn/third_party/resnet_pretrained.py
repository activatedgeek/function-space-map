import os
import flaxmodels as fm

from ..registry import register_model


@register_model
def resnet18_pt(model_dir=None, **_):
    return fm.ResNet18(
        output="logits",
        pretrained="imagenet",
        ckpt_dir=model_dir or os.environ.get("MODELDIR"),
    )


@register_model
def resnet50_pt(model_dir=None, **_):
    return fm.ResNet50(
        output="logits",
        pretrained="imagenet",
        ckpt_dir=model_dir or os.environ.get("MODELDIR"),
    )

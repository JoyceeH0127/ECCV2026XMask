# -*- coding: utf-8 -*-
from .vgg16bn import VGG16BN


def build_single_model(name="vgg16bn", pretrained=True):
    name = name.lower()
    if name in ("vgg16bn", "vgg16_bn"):
        return VGG16BN(pretrained=pretrained)
    raise ValueError(f"unknown model: {name}")


def build_model(name="vgg16bn"):
    """Build independently initialized student and teacher models.

    This intentionally mirrors ``P2RLoss/main_backup.py``: the two models are
    constructed separately and the teacher is subsequently updated from the
    student with EMA during training.
    """
    student = build_single_model(name)
    teacher = build_single_model(name)
    return student, teacher

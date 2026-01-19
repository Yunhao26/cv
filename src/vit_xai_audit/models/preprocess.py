from __future__ import annotations

from typing import Tuple

from torchvision import transforms as tvt

from vit_xai_audit.data.pil_utils import pil_to_rgb


def get_eval_transform(model_name: str):
    """
    Return an eval-time preprocessing transform compatible with the model's pretrained weights.
    Always converts PIL -> RGB first (pickle-safe for Windows workers).
    """
    model_name = model_name.lower()
    to_rgb = tvt.Lambda(pil_to_rgb)

    if model_name in {"vit_b16", "vit_base_patch16_224"}:
        import timm
        from timm.data import create_transform, resolve_model_data_config

        m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
        cfg = resolve_model_data_config(m)
        tf = create_transform(**cfg, is_training=False)
        return tvt.Compose([to_rgb, tf])

    if model_name == "resnet50":
        from torchvision.models import ResNet50_Weights

        tf = ResNet50_Weights.IMAGENET1K_V2.transforms()
        return tvt.Compose([to_rgb, tf])

    raise ValueError(f"unknown model_name for preprocess: {model_name}")


def get_input_size(model_name: str) -> int:
    model_name = model_name.lower()
    if model_name in {"vit_b16", "vit_base_patch16_224"}:
        return 224
    if model_name == "resnet50":
        return 224
    raise ValueError(f"unknown model_name: {model_name}")


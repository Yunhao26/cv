from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    name: str  # "resnet50" | "vit_b16"
    num_classes: int = 2


def build_model(spec: ModelSpec) -> nn.Module:
    name = spec.name.lower()
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, spec.num_classes)
        return m

    if name in {"vit_b16", "vit_base_patch16_224"}:
        import timm

        m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=spec.num_classes)
        return m

    raise ValueError(f"unknown model name: {spec.name}")


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)


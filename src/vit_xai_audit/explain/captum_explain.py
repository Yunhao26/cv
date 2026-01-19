from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam


def _vit_token_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    input_size: int,
) -> torch.Tensor:
    """
    Captum 0.7.0 does not support reshape_transform in LayerGradCam.
    We implement a simple token-level Grad-CAM for timm ViT:
      - hook last block output tokens [B, N, C]
      - compute channel weights from grads
      - compute token CAM and reshape to (H, W), then upsample to input_size
    Returns: [B, input_size, input_size]
    """
    acts = {}

    def fwd_hook(_m, _inp, out):
        # out: [B, N, C]
        acts["tokens"] = out
        out.retain_grad()

    handle = model.blocks[-1].register_forward_hook(fwd_hook)
    try:
        logits = model(x)  # forward, stores acts
        # gather target logit per sample and backprop
        sel = logits.gather(1, target.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        sel.backward(retain_graph=False)

        tokens = acts["tokens"]  # [B, N, C]
        grads = tokens.grad  # [B, N, C]
        if grads is None:
            raise RuntimeError("gradcam grads are None; backward hook failed")

        # weights per channel: average grad over tokens
        weights = grads.mean(dim=1)  # [B, C]
        cam = (tokens * weights.unsqueeze(1)).sum(dim=2)  # [B, N]
        cam = torch.relu(cam)

        # drop CLS token, reshape to square
        cam = cam[:, 1:]
        b, n = cam.shape
        h = w = int(np.sqrt(n))
        if h * w != n:
            raise ValueError(f"cannot reshape cam of length {n} into square map")
        cam = cam.view(b, 1, h, w)
        cam = torch.nn.functional.interpolate(cam, size=(input_size, input_size), mode="bilinear", align_corners=False)
        return cam.squeeze(1)
    finally:
        handle.remove()


@dataclass(frozen=True)
class ExplainConfig:
    ig_steps: int = 64


def _reduce_to_map(attr: torch.Tensor) -> torch.Tensor:
    # attr: [B, C, H, W] -> [B, H, W]
    return attr.abs().sum(dim=1)


@torch.no_grad()
def _predict(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    return pred, conf


def integrated_gradients_attribution(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    cfg: ExplainConfig,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)
    # Use internal_batch_size to reduce memory when n_steps is large (important on 8GB GPUs).
    attr = ig.attribute(
        x,
        baselines=baseline,
        target=target,
        n_steps=cfg.ig_steps,
        internal_batch_size=1,
    )
    return _reduce_to_map(attr)


def _vit_patch_reshape(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens: [B, N, C] (includes class token at index 0)
    returns: [B, C, H, W] for LayerGradCam
    """
    # drop cls token
    x = tokens[:, 1:, :]
    b, n, c = x.shape
    h = w = int(np.sqrt(n))
    if h * w != n:
        raise ValueError(f"cannot reshape tokens of length {n} into square map")
    x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
    return x


def grad_cam_attribution(
    model_name: str,
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    input_size: int,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    name = model_name.lower()

    if name == "resnet50":
        layer = model.layer4
        lgc = LayerGradCam(model, layer)
        attr = lgc.attribute(x, target=target)
        attr = LayerAttribution.interpolate(attr, (input_size, input_size))
        return attr.squeeze(1).abs()

    if name in {"vit_b16", "vit_base_patch16_224"}:
        attr = _vit_token_gradcam(model, x, target=target, input_size=input_size)
        return attr.abs()

    raise ValueError(f"unknown model_name for grad-cam: {model_name}")


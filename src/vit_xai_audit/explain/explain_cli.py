from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from vit_xai_audit.data.chest_xray import ChestXrayPaths
from vit_xai_audit.explain.captum_explain import ExplainConfig, grad_cam_attribution, integrated_gradients_attribution
from vit_xai_audit.models.build import ModelSpec, build_model
from vit_xai_audit.models.preprocess import get_eval_transform, get_input_size
from vit_xai_audit.utils.seed import seed_everything


def _load_ckpt(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])


@torch.no_grad()
def _pred_info(model: nn.Module, x: torch.Tensor):
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    return pred, conf, probs[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50", "vit_b16"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="./artifacts/attributions")
    ap.add_argument("--subset_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ig_steps", type=int, default=16)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: nn.Module = build_model(ModelSpec(name=args.model, num_classes=2)).to(device).eval()
    ckpt_path = Path(args.ckpt)
    _load_ckpt(model, ckpt_path, device)

    input_size = get_input_size(args.model)
    eval_tf = get_eval_transform(args.model)

    # Build test dataset (ImageFolder) with eval transform
    data_root = Path(args.data_root)
    paths = ChestXrayPaths(root=data_root / "chest_xray")
    test_ds = torch.utils.data.dataset.Dataset
    from torchvision.datasets import ImageFolder

    test_ds = ImageFolder(paths.test_dir, transform=eval_tf)

    n = min(args.subset_size, len(test_ds))
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(test_ds), size=n, replace=False).tolist()

    # keep file paths for later failure-case reporting
    img_paths = [test_ds.samples[i][0] for i in sel]
    y_true = np.array([test_ds.samples[i][1] for i in sel], dtype=np.int64)

    subset = torch.utils.data.Subset(test_ds, sel)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    cfg = ExplainConfig(ig_steps=args.ig_steps)

    attrs_ig: List[np.ndarray] = []
    attrs_gc: List[np.ndarray] = []
    y_pred_all: List[int] = []
    conf_all: List[float] = []
    p1_all: List[float] = []

    idx0 = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred, conf, p1 = _pred_info(model, x)
        target = pred  # explain the model's own decision

        ig_map = integrated_gradients_attribution(model, x, target=target, cfg=cfg)
        gc_map = grad_cam_attribution(args.model, model, x, target=target, input_size=input_size)

        attrs_ig.append(ig_map.detach().float().cpu().numpy())
        attrs_gc.append(gc_map.detach().float().cpu().numpy())

        y_pred_all.extend(pred.detach().cpu().tolist())
        conf_all.extend(conf.detach().cpu().tolist())
        p1_all.extend(p1.detach().cpu().tolist())

        idx0 += x.size(0)

        # prevent GPU memory fragmentation / accumulation
        del x, y, pred, conf, p1, target, ig_map, gc_map
        if device.type == "cuda":
            torch.cuda.empty_cache()

    attrs_ig_np = np.concatenate(attrs_ig, axis=0).astype(np.float32)
    attrs_gc_np = np.concatenate(attrs_gc, axis=0).astype(np.float32)
    y_pred_np = np.array(y_pred_all, dtype=np.int64)
    conf_np = np.array(conf_all, dtype=np.float32)
    p1_np = np.array(p1_all, dtype=np.float32)

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, object] = {
        "model": args.model,
        "ckpt": str(ckpt_path),
        "subset_size": int(n),
        "seed": int(args.seed),
        "class_to_idx": getattr(test_ds, "class_to_idx", None),
        "classes": getattr(test_ds, "classes", None),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    np.savez_compressed(
        out_dir / "attributions_test_subset.npz",
        indices=np.array(sel, dtype=np.int64),
        img_paths=np.array(img_paths, dtype=object),
        y_true=y_true,
        y_pred=y_pred_np,
        conf=conf_np,
        p1=p1_np,
        ig=attrs_ig_np,
        grad_cam=attrs_gc_np,
    )

    print("DONE", str(out_dir))


if __name__ == "__main__":
    main()


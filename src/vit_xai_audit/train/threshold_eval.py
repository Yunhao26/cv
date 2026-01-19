from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vit_xai_audit.data.chest_xray import build_datasets
from vit_xai_audit.models.build import ModelSpec, build_model


@torch.no_grad()
def collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p1 = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p1)
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def best_threshold(probs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    # predict pneumonia (class 1) if p1 >= thr else normal (0)
    best_thr = 0.5
    best_acc = -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (probs >= thr).astype(np.int64)
        acc = (pred == ys).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best_thr = float(thr)
    return best_thr, best_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.ckpt")
    ap.add_argument("--model", required=True, choices=["resnet50", "vit_b16"])
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    run_dir = ckpt_path.parent

    train_ds, val_ds, test_ds = build_datasets(
        data_root=args.data_root,
        image_size=args.image_size,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(ModelSpec(name=args.model, num_classes=2)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_probs, val_y = collect_probs(model, val_loader, device)
    thr, val_acc_thr = best_threshold(val_probs, val_y)

    test_probs, test_y = collect_probs(model, test_loader, device)
    test_pred = (test_probs >= thr).astype(np.int64)
    test_acc_thr = float((test_pred == test_y).mean())

    out: Dict[str, float] = {
        "threshold": float(thr),
        "val_acc_thresholded": float(val_acc_thr),
        "test_acc_thresholded": float(test_acc_thr),
    }
    (run_dir / "threshold_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK", json.dumps(out))


if __name__ == "__main__":
    main()


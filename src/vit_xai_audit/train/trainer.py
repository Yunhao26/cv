from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_xai_audit.train.metrics import accuracy_from_logits


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    num_workers: int
    amp: bool = True
    class_weights: Optional[Tuple[float, float]] = None


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def make_run_dir(runs_dir: str | Path, run_name: str) -> Path:
    p = Path(runs_dir) / run_name / _now_tag()
    p.mkdir(parents=True, exist_ok=False)
    return p


def build_loaders(train_ds, val_ds, batch_size: int, num_workers: int, sampler=None):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        acc = accuracy_from_logits(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    device: torch.device,
    cfg: TrainConfig,
    out_dir: Path,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "best.ckpt"

    model.to(device)
    weight_tensor = None
    if cfg.class_weights is not None:
        weight_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val_acc = -1.0
    best: Dict[str, float] = {"best_val_acc": best_val_acc}

    with log_path.open("a", encoding="utf-8") as f:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            t0 = time.time()

            running_loss = 0.0
            running_acc = 0.0
            n = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    logits = model(x)
                    loss = loss_fn(logits, y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                acc = accuracy_from_logits(logits.detach(), y)
                bs = x.size(0)
                running_loss += loss.item() * bs
                running_acc += acc * bs
                n += bs

            train_loss = running_loss / max(1, n)
            train_acc = running_acc / max(1, n)
            val_loss, val_acc = evaluate(model, val_loader, device)

            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seconds": round(time.time() - t0, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best = {"best_val_acc": best_val_acc, "best_epoch": epoch}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "best": best,
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )

    # Evaluate best checkpoint on test (if provided)
    if test_loader is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_acc = evaluate(model, test_loader, device)
        best["test_loss"] = float(test_loss)
        best["test_acc"] = float(test_acc)
        (out_dir / "summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    return best


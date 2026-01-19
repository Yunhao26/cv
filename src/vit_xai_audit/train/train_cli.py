from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset, WeightedRandomSampler

from vit_xai_audit.data.chest_xray import build_datasets
from vit_xai_audit.models.build import ModelSpec, build_model
from vit_xai_audit.train.trainer import TrainConfig, build_loaders, make_run_dir, train
from vit_xai_audit.utils.config import load_yaml
from vit_xai_audit.utils.device import DeviceConfig, apply_device_config
from vit_xai_audit.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to yaml config")
    ap.add_argument("--model", required=True, choices=["resnet50", "vit_b16"])
    ap.add_argument("--runs_dir", default="./runs")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    amp = bool(cfg.get("amp", True))
    tf32 = bool(cfg.get("tf32", True))
    apply_device_config(DeviceConfig(amp=amp, tf32=tf32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg.get("data_root", "./data")
    image_size = int(cfg.get("image_size", 224))
    model = build_model(ModelSpec(name=args.model, num_classes=2))

    # Build transforms: use model-native preprocessing for better accuracy.
    to_rgb = None
    from torchvision import transforms as tvt
    from vit_xai_audit.data.pil_utils import pil_to_rgb

    to_rgb = tvt.Lambda(pil_to_rgb)

    train_tf = None
    eval_tf = None
    if args.model == "vit_b16":
        from timm.data import create_transform, resolve_model_data_config

        data_cfg = resolve_model_data_config(model)
        train_tf = create_transform(**data_cfg, is_training=True)
        eval_tf = create_transform(**data_cfg, is_training=False)
        train_tf = tvt.Compose([to_rgb, train_tf])
        eval_tf = tvt.Compose([to_rgb, eval_tf])

    elif args.model == "resnet50":
        from torchvision.models import ResNet50_Weights

        # weights.transforms includes resize/crop/normalize
        eval_tf = ResNet50_Weights.IMAGENET1K_V2.transforms()
        train_tf = eval_tf
        train_tf = tvt.Compose([to_rgb, train_tf])
        eval_tf = tvt.Compose([to_rgb, eval_tf])

    train_ds, val_ds, test_ds = build_datasets(
        data_root=data_root,
        image_size=image_size,
        seed=seed,
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        train_transform=train_tf,
        eval_transform=eval_tf,
    )

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 8))

    sampler = None
    if bool(cfg.get("balanced_sampler", True)) and isinstance(train_ds, Subset):
        base_targets = getattr(train_ds.dataset, "targets", None)
        if base_targets is not None:
            idx = list(train_ds.indices)
            y = torch.tensor([base_targets[i] for i in idx], dtype=torch.long)
            counts = torch.bincount(y, minlength=2).float()
            class_w = (1.0 / counts).tolist()
            sample_w = torch.tensor([class_w[int(t)] for t in y.tolist()], dtype=torch.double)
            sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)

    train_loader, val_loader = build_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )
    test_loader = build_loaders(
        train_ds=test_ds,
        val_ds=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )[1]

    # Optional class weights (can be disabled if using balanced sampler)
    cw = None
    if bool(cfg.get("use_class_weights", False)) and isinstance(train_ds, Subset):
        base_targets = getattr(train_ds.dataset, "targets", None)
        if base_targets is not None:
            idx = list(train_ds.indices)
            y = torch.tensor([base_targets[i] for i in idx], dtype=torch.long)
            counts = torch.bincount(y, minlength=2).float()
            total = counts.sum().item()
            w0 = total / (2.0 * counts[0].item())
            w1 = total / (2.0 * counts[1].item())
            cw = (float(w0), float(w1))

    train_cfg = TrainConfig(
        epochs=int(cfg.get("epochs", 8)),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 8)),
        amp=amp,
        class_weights=cw,
    )

    run_dir = make_run_dir(args.runs_dir, run_name=args.model)
    (run_dir / "config_used.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    best = train(model, train_loader, val_loader, test_loader=test_loader, device=device, cfg=train_cfg, out_dir=run_dir)
    print("DONE", best, "run_dir=", str(run_dir))


if __name__ == "__main__":
    main()


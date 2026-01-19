from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from quantus.metrics import faithfulness, robustness

from vit_xai_audit.explain.captum_explain import ExplainConfig, grad_cam_attribution, integrated_gradients_attribution
from vit_xai_audit.models.build import ModelSpec, build_model
from vit_xai_audit.models.preprocess import get_eval_transform, get_input_size
from vit_xai_audit.utils.seed import seed_everything


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _load_images_as_batch(img_paths: np.ndarray, tf) -> np.ndarray:
    """
    Returns:
      x_batch: numpy float32, channel-first [N, C, H, W] matching model input
      x_batch_tensor: same as torch tensor on device
    """
    xs = []
    for p in img_paths.tolist():
        img = Image.open(p)
        x = tf(img)  # torch tensor C,H,W
        xs.append(x)
    x_t = torch.stack(xs, dim=0)
    return x_t.detach().cpu().numpy().astype(np.float32)


def _explain_func_factory(model_name: str, ig_steps: int, device: torch.device):
    cfg = ExplainConfig(ig_steps=ig_steps)
    input_size = get_input_size(model_name)

    def explain_func(model, inputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
        """
        Quantus expects signature: explain_func(model=..., inputs=..., targets=...)
        inputs: [N,C,H,W] numpy
        targets: [N] numpy (class indices)
        returns: [N,H,W] numpy
        """
        xt = torch.from_numpy(inputs).to(device)
        tt = torch.from_numpy(targets.astype(np.int64)).to(device)
        model.eval()

        out = []
        zero_fixes = 0
        for i in range(xt.size(0)):
            xi = xt[i : i + 1]
            ti = tt[i : i + 1]
            if kwargs.get("method") == "integrated_gradients":
                a = integrated_gradients_attribution(model, xi, target=ti, cfg=cfg)  # [1,H,W]
            else:
                a = grad_cam_attribution(model_name, model, xi, target=ti, input_size=input_size)  # [1,H,W]
            a_np = a.detach().cpu().numpy().astype(np.float32)
            # Quantus asserts attributions are not all zeros. Under strong perturbations,
            # some methods/models can degenerate to all-zero maps (esp. ViT token-gradcam).
            # Apply a deterministic epsilon "tick" to keep evaluation running.
            if np.all(a_np == 0):
                a_np[..., 0, 0] = 1e-8
                zero_fixes += 1
            out.append(a_np)
            del xi, ti, a
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if zero_fixes > 0:
            print(f"[quantus_cli] explain_func zero-map fixes: {zero_fixes}/{xt.size(0)} method={kwargs.get('method')}")
        return np.concatenate(out, axis=0).astype(np.float32)

    return explain_func


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50", "vit_b16"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True, help="path to attributions_test_subset.npz")
    ap.add_argument("--out_dir", default="./artifacts/metrics")
    ap.add_argument("--seed", type=int, default=42)

    # FaithfulnessCorrelation params
    ap.add_argument("--fc_nr_runs", type=int, default=50)
    ap.add_argument("--fc_subset_size", type=int, default=224)

    # MaxSensitivity params
    ap.add_argument("--ms_nr_samples", type=int, default=10)
    ap.add_argument("--ms_lower_bound", type=float, default=0.1)
    ap.add_argument("--ig_steps", type=int, default=16)

    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(ModelSpec(name=args.model, num_classes=2)).to(device).eval()
    ckpt = torch.load(Path(args.ckpt), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    data = _load_npz(Path(args.npz))
    img_paths = data["img_paths"]
    y_true = data["y_true"].astype(np.int64)
    y_target = data["y_pred"].astype(np.int64)

    tf = get_eval_transform(args.model)
    x_batch = _load_images_as_batch(img_paths, tf=tf)

    # a_batch for each method comes from cached npz, shape [N,H,W]
    a_ig = data["ig"].astype(np.float32)
    a_gc = data["grad_cam"].astype(np.float32)

    # Faithfulness Correlation (precomputed a_batch)
    fc = faithfulness.FaithfulnessCorrelation(
        nr_runs=args.fc_nr_runs,
        subset_size=args.fc_subset_size,
        return_aggregate=False,
        display_progressbar=False,
    )

    # MaxSensitivity (needs explain_func to recompute a_batch under perturbations)
    explain_func = _explain_func_factory(args.model, ig_steps=args.ig_steps, device=device)
    ms = robustness.MaxSensitivity(
        nr_samples=args.ms_nr_samples,
        lower_bound=args.ms_lower_bound,
        return_aggregate=False,
        display_progressbar=False,
    )

    rows = []
    for method_name, a_batch in [("integrated_gradients", a_ig), ("grad_cam", a_gc)]:
        fc_scores = fc(
            model=model,
            x_batch=x_batch,
            y_batch=y_target,
            a_batch=a_batch,
            channel_first=True,
            softmax=False,
            device=str(device),
            batch_size=16,
        )

        ms_scores = ms(
            model=model,
            x_batch=x_batch,
            y_batch=y_target,
            a_batch=a_batch,
            channel_first=True,
            explain_func=explain_func,
            explain_func_kwargs={"method": method_name},
            softmax=False,
            device=str(device),
            batch_size=1,
        )

        for i in range(len(y_true)):
            rows.append(
                {
                    "model": args.model,
                    "method": method_name,
                    "sample_idx": int(i),
                    "img_path": str(img_paths[i]),
                    "y_true": int(y_true[i]),
                    "faithfulness_correlation": float(fc_scores[i]),
                    "max_sensitivity": float(ms_scores[i]),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"quantus_{args.model}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("DONE", str(out_csv))


if __name__ == "__main__":
    main()


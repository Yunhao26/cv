from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _norm_map(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    a = a - np.nanmin(a)
    denom = np.nanmax(a)
    if denom <= 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return (a / denom).astype(np.float32)


def _central_mass(a: np.ndarray, frac: float = 0.6) -> float:
    """
    Heuristic for 'looks like lungs': how much attribution mass is concentrated
    in the central region of the image (lungs are typically central).
    """
    a = np.maximum(a, 0.0)
    s = float(a.sum())
    if s <= 1e-12:
        return 0.0
    h, w = a.shape
    dh = int(round(h * frac))
    dw = int(round(w * frac))
    y0 = (h - dh) // 2
    x0 = (w - dw) // 2
    return float(a[y0 : y0 + dh, x0 : x0 + dw].sum() / s)


@dataclass(frozen=True)
class SelectConfig:
    k: int = 12
    low_q: float = 0.15
    high_q: float = 0.85
    central_hi: float = 0.7
    central_lo: float = 0.3


def _select_cases(
    df: pd.DataFrame, method: str, cfg: SelectConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      intuitive_but_lowfaith: central_mass high, faithfulness low
      unintuitive_but_highfaith: central_mass low, faithfulness high
    """
    d = df[df["method"] == method].copy()
    if d.empty:
        return d, d

    low_thr = float(d["faithfulness_correlation"].quantile(cfg.low_q))
    high_thr = float(d["faithfulness_correlation"].quantile(cfg.high_q))

    intuitive_low = (
        d[(d["central_mass"] >= cfg.central_hi) & (d["faithfulness_correlation"] <= low_thr)]
        .sort_values(["faithfulness_correlation", "central_mass"], ascending=[True, False])
        .head(cfg.k)
    )
    unintuitive_high = (
        d[(d["central_mass"] <= cfg.central_lo) & (d["faithfulness_correlation"] >= high_thr)]
        .sort_values(["faithfulness_correlation", "central_mass"], ascending=[False, True])
        .head(cfg.k)
    )
    return intuitive_low, unintuitive_high


def _make_panel(
    rows: pd.DataFrame,
    out_path: Path,
    title: str,
    cmap: str = "inferno",
) -> None:
    if rows.empty:
        return

    n = len(rows)
    cols = 4
    r = int(np.ceil(n / cols))

    fig, axes = plt.subplots(r, cols, figsize=(cols * 4.2, r * 4.2))
    axes = np.array(axes).reshape(r, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for i, row in enumerate(rows.itertuples(index=False)):
        ax = axes.ravel()[i]
        img = Image.open(row.img_path).convert("RGB")
        a = np.array(row.attr_map, dtype=np.float32)
        a = _norm_map(a)

        ax.imshow(img)
        ax.imshow(a, cmap=cmap, alpha=0.45, vmin=0.0, vmax=1.0)
        ax.set_title(
            f"y={row.y_true}  pred={row.y_pred}\n"
            f"faith={row.faithfulness_correlation:.3f}  sens={row.max_sensitivity:.3f}\n"
            f"center={row.central_mass:.2f}",
            fontsize=9,
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50", "vit_b16"])
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", default="./artifacts/figures")
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / "failure_cases" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    npz = _load_npz(Path(args.npz))

    # Align rows by sample_idx (both come from same subset ordering in our pipeline)
    # metrics_csv sample_idx is 0..N-1 for each method; npz arrays are length N
    img_paths = npz["img_paths"].astype(object)
    y_true = npz["y_true"].astype(int)
    y_pred = npz["y_pred"].astype(int)

    # Expand df with image metadata
    df["img_path"] = df["sample_idx"].apply(lambda i: str(img_paths[int(i)]))
    df["y_true"] = df["sample_idx"].apply(lambda i: int(y_true[int(i)]))
    df["y_pred"] = df["sample_idx"].apply(lambda i: int(y_pred[int(i)]))

    # Add attribution maps + central_mass
    def get_attr(method: str, i: int) -> np.ndarray:
        if method == "integrated_gradients":
            return npz["ig"][i]
        return npz["grad_cam"][i]

    df["attr_map"] = df.apply(lambda r: get_attr(r["method"], int(r["sample_idx"])), axis=1)
    df["central_mass"] = df["attr_map"].apply(lambda a: _central_mass(_norm_map(a)))

    cfg = SelectConfig(k=args.k)

    for method in ["integrated_gradients", "grad_cam"]:
        intuitive_low, unintuitive_high = _select_cases(df, method=method, cfg=cfg)

        # Save case tables
        intuitive_low.drop(columns=["attr_map"]).to_csv(out_dir / f"{method}_intuitive_lowfaith.csv", index=False)
        unintuitive_high.drop(columns=["attr_map"]).to_csv(out_dir / f"{method}_unintuitive_highfaith.csv", index=False)

        # Panels
        _make_panel(
            intuitive_low,
            out_path=out_dir / f"{method}_panel_intuitive_lowfaith.png",
            title=f"{args.model} | {method} | intuitive-but-low-faithfulness",
        )
        _make_panel(
            unintuitive_high,
            out_path=out_dir / f"{method}_panel_unintuitive_highfaith.png",
            title=f"{args.model} | {method} | unintuitive-but-high-faithfulness",
        )

    print("DONE", str(out_dir))


if __name__ == "__main__":
    main()


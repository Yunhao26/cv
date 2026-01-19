from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def bootstrap_ci(x: np.ndarray, n: int = 2000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n):
        samp = rng.choice(x, size=len(x), replace=True)
        means.append(float(np.mean(samp)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="./artifacts/metrics")
    ap.add_argument("--out_dir", default="./artifacts/figures/summary")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        metrics_dir / "quantus_resnet50.csv",
        metrics_dir / "quantus_vit_b16.csv",
    ]
    dfs = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"missing metrics file: {f}")
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    # Clean up pathological values from Quantus runtime warnings (e.g., division by zero -> inf).
    for col in ["faithfulness_correlation", "max_sensitivity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan

    # Aggregate stats per (model, method)
    rows = []
    for (model, method), g in df.groupby(["model", "method"]):
        for metric in ["faithfulness_correlation", "max_sensitivity"]:
            x = g[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "metric": metric,
                    "n": int(np.sum(~np.isnan(x))),
                    "mean": float(np.nanmean(x)),
                    "median": float(np.nanmedian(x)),
                    "std": float(np.nanstd(x)),
                    "ci95_lo": bootstrap_ci(x, seed=args.seed)[0],
                    "ci95_hi": bootstrap_ci(x, seed=args.seed)[1],
                }
            )

    summary = pd.DataFrame(rows).sort_values(["metric", "model", "method"])
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    # Plots
    sns.set_theme(style="whitegrid")

    # Boxplots: Faithfulness
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(
        data=df,
        x="model",
        y="faithfulness_correlation",
        hue="method",
        showfliers=False,
    )
    plt.title("Faithfulness Correlation (Quantus)")
    plt.tight_layout()
    plt.savefig(out_dir / "faithfulness_boxplot.png", dpi=180)
    plt.close()

    # Boxplots: MaxSensitivity
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(
        data=df,
        x="model",
        y="max_sensitivity",
        hue="method",
        showfliers=False,
    )
    plt.title("Max-Sensitivity (Quantus)")
    plt.tight_layout()
    plt.savefig(out_dir / "max_sensitivity_boxplot.png", dpi=180)
    plt.close()

    # Mean+CI bar chart per metric
    for metric in ["faithfulness_correlation", "max_sensitivity"]:
        sub = summary[summary["metric"] == metric].copy()
        sub["err_lo"] = sub["mean"] - sub["ci95_lo"]
        sub["err_hi"] = sub["ci95_hi"] - sub["mean"]

        plt.figure(figsize=(9, 4.5))
        # Create a stable order
        order = ["resnet50", "vit_b16"]
        hue_order = ["integrated_gradients", "grad_cam"]

        # draw bars manually
        x_positions = []
        heights = []
        yerr = [[], []]
        labels = []
        xpos = 0
        width = 0.35
        for m in order:
            for j, meth in enumerate(hue_order):
                row = sub[(sub["model"] == m) & (sub["method"] == meth)]
                if row.empty:
                    val = np.nan
                    lo = hi = np.nan
                else:
                    val = float(row["mean"].iloc[0])
                    lo = float(row["err_lo"].iloc[0])
                    hi = float(row["err_hi"].iloc[0])
                x_positions.append(xpos + j * width)
                heights.append(val)
                yerr[0].append(lo)
                yerr[1].append(hi)
                labels.append(f"{m}\n{meth}")
            xpos += 1.2

        plt.bar(x_positions, heights, yerr=yerr, capsize=4)
        plt.xticks(x_positions, labels, rotation=0, fontsize=8)
        plt.title(f"Mean ± 95% CI: {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_mean_ci.png", dpi=180)
        plt.close()

    print("DONE", str(out_dir))


if __name__ == "__main__":
    main()


#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="outputs/results")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = []

    for p in sorted(results_dir.glob("*/metrics.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": p.parent.name,
                "method": payload["method"],
                "accuracy": payload["metrics"]["accuracy"],
                "auroc": payload["metrics"]["auroc"],
                "f1": payload["metrics"]["f1"],
                "trainable_params": payload["resources"]["trainable_params"],
                "train_time_sec": payload["timing"]["train_time_sec"],
            }
        )

    if not rows:
        raise RuntimeError(f"No metrics.json files found under {results_dir}")

    df = pd.DataFrame(rows).sort_values(by=["method", "run"])
    out_csv = results_dir / "comparison_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved comparison table: {out_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df["trainable_params"], df["auroc"])
    axes[0].set_xlabel("Trainable Parameters")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("AUROC vs Trainable Parameters")

    axes[1].scatter(df["train_time_sec"], df["auroc"])
    axes[1].set_xlabel("Training Time (sec)")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("AUROC vs Training Time")

    for _, row in df.iterrows():
        axes[0].annotate(row["method"], (row["trainable_params"], row["auroc"]))
        axes[1].annotate(row["method"], (row["train_time_sec"], row["auroc"]))

    fig.tight_layout()
    out_png = results_dir / "comparison_plots.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved comparison plot: {out_png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Create simple comparison plots from experiment_summary.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    summary_path = Path("outputs/experiment_summary.csv")
    if not summary_path.exists():
        raise FileNotFoundError("Run experiments first: outputs/experiment_summary.csv not found")

    df = pd.read_csv(summary_path)
    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1) AUROC vs trainable params
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="trainable_params", y="auroc", hue="method", style="method", s=120)
    plt.xscale("log")
    plt.title("AUROC vs Trainable Parameters")
    plt.tight_layout()
    plt.savefig(plot_dir / "auroc_vs_params.png", dpi=200)
    plt.close()

    # 2) F1 vs runtime
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="train_seconds", y="f1", hue="method", style="method", s=120)
    plt.title("F1 vs Training Time")
    plt.tight_layout()
    plt.savefig(plot_dir / "f1_vs_runtime.png", dpi=200)
    plt.close()

    # 3) Bar chart for key metrics
    long_df = df.melt(id_vars=["method", "lora_rank"], value_vars=["accuracy", "auroc", "f1"], var_name="metric")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=long_df, x="method", y="value", hue="metric")
    plt.title("Method Comparison on Test Set")
    plt.tight_layout()
    plt.savefig(plot_dir / "metric_bars.png", dpi=200)
    plt.close()

    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()

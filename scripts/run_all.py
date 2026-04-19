#!/usr/bin/env python
from __future__ import annotations

import subprocess

EXPERIMENT_CONFIGS = [
    "configs/experiments/zero_shot.yaml",
    "configs/experiments/linear_probe.yaml",
    "configs/experiments/lora_r4.yaml",
    "configs/experiments/lora_r8.yaml",
    "configs/experiments/lora_r16.yaml",
    "configs/experiments/full_finetune.yaml",
]


def main():
    for cfg in EXPERIMENT_CONFIGS:
        print(f"\n=== Running: {cfg} ===")
        cmd = ["python", "scripts/run_experiment.py", "--config", cfg]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

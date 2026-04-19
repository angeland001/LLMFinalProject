#!/usr/bin/env python
"""Convenience launcher for all experiment variants."""

from __future__ import annotations

import subprocess

LORA_RANKS = [4, 8, 16]


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    base = ["python", "scripts/run_experiment.py", "--config", "configs/base.yaml"]

    run(base + ["--method", "zero_shot"])
    run(base + ["--method", "linear_probe"])

    for rank in LORA_RANKS:
        run(base + ["--method", "lora", "--lora-rank", str(rank)])

    run(base + ["--method", "full_finetune"])


if __name__ == "__main__":
    main()

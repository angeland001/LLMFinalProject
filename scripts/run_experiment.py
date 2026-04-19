#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.data.rsna_dataset import make_rsna_dataloaders
from src.models.clip_setup import load_clip
from src.training.common import get_device, get_peak_memory_gb
from src.training.full_finetune import run_full_finetune
from src.training.linear_probe import run_linear_probe
from src.training.lora_finetune import run_lora_finetune
from src.training.zero_shot import run_zero_shot
from src.utils.logging_utils import make_run_dir, save_json, save_yaml
from src.utils.params import count_total_params, count_trainable_params, trainable_ratio
from src.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    processor, clip_model = load_clip(cfg["model"]["backbone"])
    loaders = make_rsna_dataloaders(
        csv_path=cfg["data"]["csv_path"],
        processor=processor,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )

    method = cfg["train"]["method"]
    run_dir = make_run_dir(cfg["project"]["output_root"], method)

    if method == "zero_shot":
        metrics = run_zero_shot(
            clip_model=clip_model,
            processor=processor,
            data_loader=loaders["test"],
            prompts=cfg["zero_shot"]["prompts"],
            device=device,
        )
        timing = {"train_time_sec": 0.0}
        history = {}
        model_for_params = clip_model

    elif method == "linear_probe":
        model_for_params, history, timing, metrics = run_linear_probe(clip_model, loaders, cfg, device)

    elif method == "lora":
        model_for_params, history, timing, metrics = run_lora_finetune(clip_model, loaders, cfg, device)

    elif method == "full_finetune":
        model_for_params, history, timing, metrics = run_full_finetune(clip_model, loaders, cfg, device)

    else:
        raise ValueError(f"Unknown method: {method}")

    result = {
        "method": method,
        "metrics": metrics,
        "history": history,
        "timing": timing,
        "resources": {
            "peak_gpu_mem_gb": get_peak_memory_gb(device),
            "total_params": count_total_params(model_for_params),
            "trainable_params": count_trainable_params(model_for_params),
            "trainable_ratio_percent": trainable_ratio(model_for_params),
        },
    }

    save_yaml(Path(run_dir) / "config_resolved.yaml", cfg)
    save_json(Path(run_dir) / "metrics.json", result)

    summary = (
        f"Method: {method}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"AUROC: {metrics['auroc']:.4f}\n"
        f"F1: {metrics['f1']:.4f}\n"
        f"Trainable params: {result['resources']['trainable_params']}\n"
        f"Trainable ratio (%): {result['resources']['trainable_ratio_percent']:.6f}\n"
        f"Peak GPU mem (GB): {result['resources']['peak_gpu_mem_gb']:.4f}\n"
        f"Train time (sec): {timing['train_time_sec']:.2f}\n"
    )
    (Path(run_dir) / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)


if __name__ == "__main__":
    main()

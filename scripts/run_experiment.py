#!/usr/bin/env python
"""Run one of: zero_shot, linear_probe, lora, full_finetune."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.rsna_dataset import make_dataloaders
from src.eval.zero_shot import run_zero_shot
from src.models.clip_wrapper import (
    CLIPBinaryClassifier,
    apply_lora_to_clip_vision,
    build_clip_backbone,
    freeze_all,
)
from src.trainers.train import train_binary_classifier
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.metrics import count_trainable_params
from src.utils.reproducibility import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--method", type=str, required=True, choices=["zero_shot", "linear_probe", "lora", "full_finetune"])
    parser.add_argument("--lora-rank", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))

    processor, clip_model, hidden_size = build_clip_backbone(cfg["model"]["clip_model_name"])
    train_loader, val_loader, test_loader = make_dataloaders(cfg, processor)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result_dir = ensure_dir(Path(cfg["project"]["output_dir"]) / args.method / run_id)

    if args.method == "zero_shot":
        metrics = run_zero_shot(
            model=clip_model,
            processor=processor,
            dataloader=test_loader,
            class_names=cfg["data"]["class_names"],
            prompt_templates=cfg["data"]["prompt_templates"],
            device=device,
        )
        result = {
            "method": args.method,
            "metrics": metrics,
            "trainable_params": 0,
            "total_params": sum(p.numel() for p in clip_model.parameters()),
            "trainable_pct": 0.0,
            "train_seconds": 0.0,
            "max_gpu_mem_mb": None,
        }
    else:
        if args.method == "linear_probe":
            freeze_all(clip_model)
            model = CLIPBinaryClassifier(clip_model=clip_model, hidden_size=hidden_size)
            # Only linear head trainable
            for p in model.classifier.parameters():
                p.requires_grad = True
            lr = cfg["experiments"]["linear_probe"]["lr"]
            epochs = cfg["experiments"]["linear_probe"]["epochs"]

        elif args.method == "lora":
            freeze_all(clip_model)
            lora_cfg = cfg["experiments"]["lora"]
            clip_model = apply_lora_to_clip_vision(
                model=clip_model,
                rank=args.lora_rank,
                alpha=lora_cfg["alpha"],
                dropout=lora_cfg["dropout"],
                target_modules=lora_cfg["target_modules"],
            )
            model = CLIPBinaryClassifier(clip_model=clip_model, hidden_size=hidden_size)
            for p in model.classifier.parameters():
                p.requires_grad = True
            lr = cfg["train"]["lr"]
            epochs = cfg["train"]["epochs"]

        elif args.method == "full_finetune":
            model = CLIPBinaryClassifier(clip_model=clip_model, hidden_size=hidden_size)
            lr = cfg["train"]["lr"]
            epochs = cfg["train"]["epochs"]

        trainable, total, pct = count_trainable_params(model)

        train_out = train_binary_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=cfg["train"]["weight_decay"],
            grad_clip_norm=cfg["train"]["gradient_clip_norm"],
        )

        result = {
            "method": args.method,
            "lora_rank": args.lora_rank if args.method == "lora" else None,
            "best_val": train_out.best_val,
            "metrics": train_out.test_metrics,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": pct,
            "train_seconds": train_out.train_seconds,
            "max_gpu_mem_mb": train_out.max_gpu_mem_mb,
        }

    save_json(result, result_dir / "metrics.json")

    # Keep an aggregate CSV that is easy to load in final analysis.
    flat = {
        "method": result["method"],
        "lora_rank": result.get("lora_rank"),
        "accuracy": result["metrics"].get("accuracy"),
        "auroc": result["metrics"].get("auroc"),
        "f1": result["metrics"].get("f1"),
        "trainable_params": result["trainable_params"],
        "trainable_pct": result["trainable_pct"],
        "train_seconds": result["train_seconds"],
        "max_gpu_mem_mb": result["max_gpu_mem_mb"],
        "result_dir": str(result_dir),
    }
    summary_path = Path(cfg["project"]["output_dir"]) / "experiment_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([flat])], ignore_index=True)
    else:
        df = pd.DataFrame([flat])
    df.to_csv(summary_path, index=False)

    print(f"Saved results to: {result_dir}")
    print(df.tail(5))


if __name__ == "__main__":
    main()

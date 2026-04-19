"""Training loop utilities for linear probe, LoRA, and full fine-tuning."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.utils.metrics import binary_metrics


@dataclass
class TrainResult:
    best_val: dict
    test_metrics: dict
    train_seconds: float
    max_gpu_mem_mb: float | None


def _run_epoch(model, dataloader, device, optimizer=None, grad_clip_norm: float | None = None):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    y_true, y_prob = [], []

    for batch in tqdm(dataloader, leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].float().to(device)

        logits = model(pixel_values)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.append(probs)
        y_true.append(labels.detach().cpu().numpy())
        losses.append(loss.item())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    metrics = binary_metrics(y_true=y_true, y_prob=y_prob)
    metrics["loss"] = float(np.mean(losses)) if losses else None
    return metrics


def train_binary_classifier(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float | None,
):
    model.to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)

    best_val = None
    best_state = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer=optimizer, grad_clip_norm=grad_clip_norm)
        val_metrics = _run_epoch(model, val_loader, device, optimizer=None)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} val auc={val_metrics['auroc']} val f1={val_metrics['f1']:.4f}"
        )

        score = val_metrics["auroc"] if val_metrics["auroc"] is not None else val_metrics["accuracy"]
        if best_val is None:
            is_better = True
        else:
            best_score = best_val["auroc"] if best_val["auroc"] is not None else best_val["accuracy"]
            is_better = score > best_score

        if is_better:
            best_val = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _run_epoch(model, test_loader, device, optimizer=None)

    max_mem = None
    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return TrainResult(
        best_val=best_val if best_val is not None else {},
        test_metrics=test_metrics,
        train_seconds=train_seconds,
        max_gpu_mem_mb=max_mem,
    )

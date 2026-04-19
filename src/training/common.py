from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.utils.metrics import compute_binary_classification_metrics


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_auroc = -1.0
    history = {"train_loss": [], "val_auroc": []}

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, leave=False):
            px = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(px)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            losses.append(loss.item())

        val_metrics = evaluate_classifier_model(model, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_auroc"].append(val_metrics["auroc"])

        if np.nan_to_num(val_metrics["auroc"], nan=-1.0) > best_val_auroc:
            best_val_auroc = np.nan_to_num(val_metrics["auroc"], nan=-1.0)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    return history, {"train_time_sec": elapsed}


@torch.no_grad()
def evaluate_classifier_model(model: nn.Module, data_loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, probs = [], []
    for batch in data_loader:
        px = batch["pixel_values"].to(device)
        y = batch["labels"].cpu().numpy()
        logits = model(px)
        p1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        ys.append(y)
        probs.append(p1)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    return compute_binary_classification_metrics(y_true, y_prob)


def get_peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3))

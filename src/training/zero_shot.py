from __future__ import annotations

import numpy as np
import torch

from src.utils.metrics import compute_binary_classification_metrics


@torch.no_grad()
def run_zero_shot(clip_model, processor, data_loader, prompts, device):
    clip_model.to(device)
    clip_model.eval()

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_feats = clip_model.get_text_features(**text_inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    ys, probs = [], []
    for batch in data_loader:
        px = batch["pixel_values"].to(device)
        y = batch["labels"].cpu().numpy()

        image_feats = clip_model.get_image_features(pixel_values=px)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        logits = image_feats @ text_feats.T
        p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        ys.append(y)
        probs.append(p)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    return compute_binary_classification_metrics(y_true, y_prob)

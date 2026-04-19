"""Zero-shot evaluation for CLIP using class prompts."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import binary_metrics


@torch.no_grad()
def run_zero_shot(model, processor, dataloader, class_names: list[str], prompt_templates: list[str], device):
    model.eval()
    model.to(device)

    # Build ensemble text embeddings per class from templates
    class_text_embeds = []
    for class_name in class_names:
        prompts = [t.format(class_name) for t in prompt_templates]
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        class_embed = text_features.mean(dim=0)
        class_embed = class_embed / class_embed.norm(dim=-1, keepdim=True)
        class_text_embeds.append(class_embed)

    text_matrix = torch.stack(class_text_embeds, dim=0)  # [num_classes, hidden]

    y_true, y_prob = [], []
    for batch in tqdm(dataloader, desc="zero-shot"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].cpu().numpy()

        image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_matrix.T
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        y_true.append(labels)
        y_prob.append(probs)

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    return binary_metrics(y_true=y_true, y_prob=y_prob)

"""Factory helpers for CLIP + classification heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor


class CLIPBinaryClassifier(nn.Module):
    """Binary classifier built on CLIP image encoder features."""

    def __init__(self, clip_model: CLIPModel, hidden_size: int):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values: torch.Tensor):
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.classifier(image_features).squeeze(-1)
        return logits


def build_clip_backbone(model_name: str):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    hidden_size = model.config.projection_dim
    return processor, model, hidden_size


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def apply_lora_to_clip_vision(model: CLIPModel, rank: int, alpha: int, dropout: float, target_modules: list[str]):
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model.vision_model = get_peft_model(model.vision_model, lora_cfg)
    return model

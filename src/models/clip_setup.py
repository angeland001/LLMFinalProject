from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import CLIPModel, CLIPProcessor


class CLIPBinaryClassifier(nn.Module):
    """CLIP vision encoder + classification head for supervised binary classification."""

    def __init__(self, clip_model: CLIPModel, num_classes: int = 2):
        super().__init__()
        self.clip_model = clip_model
        hidden_dim = clip_model.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        cls_emb = vision_outputs.pooler_output
        logits = self.classifier(cls_emb)
        return logits


def load_clip(backbone: str):
    processor = CLIPProcessor.from_pretrained(backbone)
    model = CLIPModel.from_pretrained(backbone)
    return processor, model


def freeze_clip_encoder(clip_model: CLIPModel) -> None:
    for p in clip_model.vision_model.parameters():
        p.requires_grad = False


def apply_lora_to_vision_module(vision_module: nn.Module, lora_cfg: dict):
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
    )
    peft_model = get_peft_model(vision_module, peft_config)
    # Return the inner model with LoRA layers injected directly into its Linear modules.
    # The PEFT wrapper's forward() injects inputs_embeds=None which conflicts with CLIP's
    # internal encoder call: encoder(inputs_embeds=hidden_states, **kwargs).
    return peft_model.base_model.model

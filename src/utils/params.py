from __future__ import annotations

import torch.nn as nn


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_ratio(model: nn.Module) -> float:
    total = count_total_params(model)
    trainable = count_trainable_params(model)
    return 100.0 * trainable / total if total > 0 else 0.0

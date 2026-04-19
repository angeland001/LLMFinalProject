"""Utilities for deterministic-ish experiment setup."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Note: full determinism can reduce speed and may not be available for every op.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_config: str = "auto") -> torch.device:
    """Resolve user config into an actual torch device."""
    if device_config != "auto":
        return torch.device(device_config)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

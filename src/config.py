from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml



def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `updates` into `base`."""
    merged = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load experiment config and optionally merge from base config."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "base_config" in cfg:
        base_path = Path(cfg["base_config"])
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        with base_path.open("r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_update(base_cfg, {k: v for k, v in cfg.items() if k != "base_config"})

    return cfg

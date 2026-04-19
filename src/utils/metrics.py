from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, AUROC, and F1 for binary classification."""
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = float("nan")
    return out

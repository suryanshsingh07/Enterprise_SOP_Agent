"""
Evaluation Metrics — Chest X-ray Benchmark
============================================
Additional metric computation utilities beyond sklearn defaults.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int = 6,
) -> Dict[str, float]:
    """Compute comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities, shape (N, C).
    n_classes : int
        Number of classes.

    Returns
    -------
    dict[str, float]
        Dictionary with all computed metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics["macro_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            pass

    return {k: round(v, 4) for k, v in metrics.items()}


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Labels.
    class_names : list[str]
        Class names.

    Returns
    -------
    dict[str, dict[str, float]]
        {class_name: {precision, recall, f1}, ...}
    """
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    result = {}
    for i, name in enumerate(class_names):
        if i < len(precisions):
            result[name] = {
                "precision": round(precisions[i], 4),
                "recall": round(recalls[i], 4),
                "f1": round(f1s[i], 4),
            }

    return result

"""
XAI Utilities — Chest X-ray Benchmark
========================================
Shared utilities for explainability modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


def load_and_resize(img_path: str | Path, size: int = 224) -> np.ndarray:
    """Load an image and resize to square.

    Parameters
    ----------
    img_path : str | Path
        Image path.
    size : int
        Target size.

    Returns
    -------
    np.ndarray
        RGB image (size, size, 3).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size))


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalise a heatmap to [0, 1].

    Parameters
    ----------
    heatmap : np.ndarray
        Input heatmap.

    Returns
    -------
    np.ndarray
        Normalised heatmap.
    """
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        return (heatmap - h_min) / (h_max - h_min)
    return np.zeros_like(heatmap)


def compute_attention_coverage(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute the fraction of the image covered by high attention.

    Parameters
    ----------
    heatmap : np.ndarray
        Normalised heatmap [0, 1].
    threshold : float
        Attention threshold. Default 0.5.

    Returns
    -------
    float
        Fraction of pixels above threshold.
    """
    return float((heatmap > threshold).sum() / heatmap.size)

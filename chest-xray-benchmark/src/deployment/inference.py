"""
Inference Utilities — Chest X-ray Benchmark
=============================================
Shared inference helpers for deployment and API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def load_and_preprocess(
    image_path: str | Path,
    img_size: int = 224,
) -> torch.Tensor:
    """Load an image from disk and preprocess for model input.

    Pipeline: load → grayscale → CLAHE → RGB → normalize → tensor

    Parameters
    ----------
    image_path : str | Path
        Path to the image.
    img_size : int
        Target size. Default 224.

    Returns
    -------
    torch.Tensor
        Preprocessed tensor, shape (1, 3, img_size, img_size).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (img_size, img_size))

    # Gray → RGB → normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_rgb - mean) / std

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def predict_single(
    model: nn.Module,
    image_path: str | Path,
    class_names: List[str],
    device: torch.device,
    img_size: int = 224,
) -> Dict:
    """Run inference on a single image.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    image_path : str | Path
        Image path.
    class_names : list[str]
        Class names.
    device : torch.device
        Compute device.
    img_size : int
        Input size.

    Returns
    -------
    dict
        {prediction, confidence, all_probabilities, inference_time_ms}
    """
    model.eval()
    tensor = load_and_preprocess(image_path, img_size).to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        proba = F.softmax(logits, dim=1).cpu().numpy()[0]
    inference_ms = (time.time() - t0) * 1000

    pred_id = int(np.argmax(proba))

    return {
        "prediction": class_names[pred_id],
        "prediction_id": pred_id,
        "confidence": float(proba[pred_id]),
        "all_probabilities": {
            class_names[i]: float(p) for i, p in enumerate(proba)
        },
        "inference_time_ms": round(inference_ms, 2),
    }


def predict_batch(
    model: nn.Module,
    image_paths: List[str | Path],
    class_names: List[str],
    device: torch.device,
    batch_size: int = 8,
    img_size: int = 224,
) -> List[Dict]:
    """Run inference on a batch of images.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    image_paths : list
        List of image paths.
    class_names : list[str]
        Class names.
    device : torch.device
        Compute device.
    batch_size : int
        Batch size. Default 8.
    img_size : int
        Input size.

    Returns
    -------
    list[dict]
        List of prediction dicts.
    """
    model.eval()
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []

        for path in batch_paths:
            try:
                tensor = load_and_preprocess(path, img_size)
                tensors.append(tensor)
            except Exception as e:
                results.append({"error": str(e), "path": str(path)})
                continue

        if not tensors:
            continue

        batch_tensor = torch.cat(tensors, dim=0).to(device)

        t0 = time.time()
        with torch.no_grad():
            logits = model(batch_tensor)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        batch_ms = (time.time() - t0) * 1000

        for j, p in enumerate(proba):
            pred_id = int(np.argmax(p))
            results.append({
                "prediction": class_names[pred_id],
                "prediction_id": pred_id,
                "confidence": float(p[pred_id]),
                "all_probabilities": {
                    class_names[k]: float(v) for k, v in enumerate(p)
                },
                "inference_time_ms": round(batch_ms / len(proba), 2),
            })

    return results

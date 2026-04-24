"""
Model Factory — Chest X-ray Benchmark
========================================
Centralised factory for creating any model by name with consistent interface.

Functions:
    create_model — build any model by name string
    list_available_models — list all supported model names
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch.nn as nn

logger = logging.getLogger(__name__)

# Registry of all supported models
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Classical ML ──────────────────────────────────────────────────────────
    "svm": {
        "module": "src.models.classical_models",
        "builder": "build_svm",
        "family": "classical",
    },
    "logistic_regression": {
        "module": "src.models.classical_models",
        "builder": "build_logistic_regression",
        "family": "classical",
    },
    "xgboost": {
        "module": "src.models.classical_models",
        "builder": "build_xgboost",
        "family": "ensemble",
    },
    "adaboost": {
        "module": "src.models.classical_models",
        "builder": "build_adaboost",
        "family": "ensemble",
    },

    # ── CNN ────────────────────────────────────────────────────────────────────
    "efficientnet_b3": {
        "module": "src.models.cnn_models",
        "class": "EfficientNetChest",
        "family": "cnn",
    },
    "resnet50": {
        "module": "src.models.cnn_models",
        "class": "ResNet50Chest",
        "family": "cnn",
    },
    "patch_lstm": {
        "module": "src.models.cnn_models",
        "class": "PatchLSTMChest",
        "family": "cnn",
    },

    # ── Transformer ───────────────────────────────────────────────────────────
    "swin_tiny": {
        "module": "src.models.transformer_models",
        "class": "SwinChest",
        "family": "transformer",
    },
    "vit_base16": {
        "module": "src.models.transformer_models",
        "class": "ViTChest",
        "family": "transformer",
    },
    "autoencoder": {
        "module": "src.models.transformer_models",
        "class": "ChestAutoencoder",
        "family": "transformer",
    },

    # ── Detection ─────────────────────────────────────────────────────────────
    "yolo": {
        "module": "src.models.detection_models",
        "class": "YOLODetector",
        "family": "detection",
    },
    "faster_rcnn": {
        "module": "src.models.detection_models",
        "class": "FasterRCNNChest",
        "family": "detection",
    },
    "mask_rcnn": {
        "module": "src.models.detection_models",
        "class": "MaskRCNNChest",
        "family": "detection",
    },

    # ── ANN ────────────────────────────────────────────────────────────────────
    "ann": {
        "module": "src.models.ann_model",
        "class": "ANNChest",
        "family": "classical",
    },
}


def create_model(name: str, **kwargs: Any) -> Any:
    """Create a model instance by name.

    Parameters
    ----------
    name : str
        Model name (case-insensitive). See ``list_available_models()``.
    **kwargs
        Keyword arguments passed to the model constructor or builder.

    Returns
    -------
    Any
        Model instance (nn.Module for DL, sklearn estimator for classical).

    Raises
    ------
    ValueError
        If model name is not recognised.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list_available_models()}"
        )

    config = MODEL_REGISTRY[key]

    import importlib
    mod = importlib.import_module(config["module"])

    if "class" in config:
        cls = getattr(mod, config["class"])
        return cls(**kwargs)
    elif "builder" in config:
        builder = getattr(mod, config["builder"])
        return builder(**kwargs)
    else:
        raise ValueError(f"Model '{name}' has no class or builder defined.")


def list_available_models() -> list[str]:
    """List all supported model names.

    Returns
    -------
    list[str]
        Sorted list of model identifiers.
    """
    return sorted(MODEL_REGISTRY.keys())


def get_model_family(name: str) -> str:
    """Get the model family for a given name.

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    str
        Family: 'classical', 'ensemble', 'cnn', 'transformer', 'detection'.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in MODEL_REGISTRY:
        return "unknown"
    return MODEL_REGISTRY[key]["family"]


if __name__ == "__main__":
    print("=" * 60)
    print("  Model Factory — Demo")
    print("=" * 60)

    print("\n  Available models:")
    for name in list_available_models():
        family = get_model_family(name)
        print(f"    {name:25s} │ {family}")

    # Create one of each family
    print("\n  Creating models:")
    svm = create_model("svm")
    print(f"    SVM: {type(svm).__name__}")

    enet = create_model("efficientnet_b3", n_classes=6, freeze_backbone=True)
    print(f"    EfficientNet-B3: {type(enet).__name__}")

    ann = create_model("ann", input_dim=512, n_classes=6)
    print(f"    ANN: {type(ann).__name__}")

    print("\n  ✓ Model factory demo complete.")

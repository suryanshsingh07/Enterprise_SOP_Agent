"""
Model Registry — Chest X-ray API
===================================
Loads, caches, and serves all model variants at application startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for loading and caching trained models.

    Supports:
    - Deep learning models: EfficientNet-B3, ResNet-50, Swin-Tiny, ViT-B/16
    - Classical models: SVM, Logistic Regression, XGBoost, AdaBoost

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory containing model checkpoints (.pth files).
    classical_dir : str | Path
        Directory containing classical model files (.joblib files).
    device : str
        Compute device ('cuda', 'cpu'). Default 'cuda'.
    n_classes : int
        Number of output classes. Default 6.
    """

    # Map of model name → (checkpoint filename, module path, class name)
    DL_MODELS = {
        "EfficientNet-B3": ("efficientnet_b3_best.pth", "src.models.cnn_models", "EfficientNetChest"),
        "ResNet-50": ("resnet50_best.pth", "src.models.cnn_models", "ResNet50Chest"),
        "Swin-Tiny": ("swin_tiny_best.pth", "src.models.transformer_models", "SwinChest"),
        "ViT-B/16": ("vit_base16_best.pth", "src.models.transformer_models", "ViTChest"),
    }

    CLASSICAL_MODELS = {
        "SVM": "SVM.joblib",
        "LogisticRegression": "LogisticRegression.joblib",
        "XGBoost": "XGBoost.joblib",
        "AdaBoost": "AdaBoost.joblib",
    }

    def __init__(
        self,
        checkpoint_dir: str | Path = "outputs/checkpoints",
        classical_dir: str | Path = "outputs/classical",
        device: str = "cuda",
        n_classes: int = 6,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.classical_dir = Path(classical_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes

        self._models: Dict[str, Any] = {}
        self._model_types: Dict[str, str] = {}  # name → 'dl' or 'classical'

    def load_all(self) -> None:
        """Load all available model checkpoints at startup."""
        logger.info("Loading models...")

        # Load DL models
        for name, (ckpt_file, module_path, class_name) in self.DL_MODELS.items():
            ckpt_path = self.checkpoint_dir / ckpt_file
            if ckpt_path.exists():
                try:
                    import importlib
                    mod = importlib.import_module(module_path)
                    model_cls = getattr(mod, class_name)
                    model = model_cls(n_classes=self.n_classes, freeze_backbone=False)

                    checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(self.device)
                    model.eval()

                    self._models[name] = model
                    self._model_types[name] = "dl"
                    logger.info(f"  ✓ Loaded: {name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {name}: {e}")
            else:
                logger.warning(f"  ⚠ Checkpoint not found: {ckpt_path}")

        # Load classical models
        for name, joblib_file in self.CLASSICAL_MODELS.items():
            joblib_path = self.classical_dir / joblib_file
            if joblib_path.exists():
                try:
                    import joblib
                    model = joblib.load(joblib_path)
                    self._models[name] = model
                    self._model_types[name] = "classical"
                    logger.info(f"  ✓ Loaded: {name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {name}: {e}")
            else:
                logger.warning(f"  ⚠ Classical model not found: {joblib_path}")

        logger.info(f"  Total models loaded: {len(self._models)}")

    def get_model(self, name: str) -> Any:
        """Get a loaded model by name.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        Any
            The loaded model instance.

        Raises
        ------
        KeyError
            If model not found.
        """
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not found. Available: {self.list_models()}"
            )
        return self._models[name]

    def get_model_type(self, name: str) -> str:
        """Get model type ('dl' or 'classical').

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        str
            'dl' or 'classical'.
        """
        return self._model_types.get(name, "unknown")

    def list_models(self) -> List[str]:
        """List all loaded model names.

        Returns
        -------
        list[str]
            Names of loaded models.
        """
        return list(self._models.keys())

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models

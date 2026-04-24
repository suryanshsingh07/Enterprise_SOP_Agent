"""
Grad-CAM Explainability — Chest X-ray Benchmark
=================================================
Gradient-weighted Class Activation Mapping for CNN model interpretability.

Classes:
    GradCAM — generic Grad-CAM with hooks on any target layer
Functions:
    get_target_layer — model-specific target layer resolver
    batch_gradcam — generate Grad-CAM for multiple samples
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# GradCAM
# =============================================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Registers forward and backward hooks on a target layer to capture
    activations and gradients, then computes a heatmap showing which
    image regions were most important for the prediction.

    Parameters
    ----------
    model : nn.Module
        The classification model.
    target_layer_name : str
        Name of the target layer (reachable via model.named_modules()).
    """

    def __init__(self, model: nn.Module, target_layer_name: str) -> None:
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List = []

        # Find and hook the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(
                f"Layer '{target_layer_name}' not found. "
                f"Available layers: {[n for n, _ in model.named_modules() if n]}"
            )

        # Register hooks
        self._hooks.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """Capture activations during forward pass."""
        self._activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """Capture gradients during backward pass."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        img_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image.

        Parameters
        ----------
        img_tensor : torch.Tensor
            Input image tensor, shape (1, 3, H, W) or (3, H, W).
        class_idx : int, optional
            Target class index. If None, uses predicted class.

        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W), normalised to [0, 1].
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device).requires_grad_(True)

        # Forward pass
        output = self.model(img_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward from the target class
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        # Get captured activations and gradients
        activations = self._activations  # (1, C, h, w)
        gradients = self._gradients      # (1, C, h, w)

        if activations is None or gradients is None:
            logger.warning("No activations/gradients captured. Check target layer.")
            return np.zeros((img_tensor.shape[2], img_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients → channel weights
        # alpha_k = (1/hw) * sum(dY_c / dA_k)
        alpha = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (alpha * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def visualise(
        self,
        img_path: str | Path,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        class_name: str = "",
        confidence: float = 0.0,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Create 3-panel visualisation: Original | Heatmap | Overlay.

        Parameters
        ----------
        img_path : str | Path
            Path to the original image.
        heatmap : np.ndarray
            Grad-CAM heatmap (H, W), [0, 1].
        alpha : float
            Overlay alpha blending. Default 0.4.
        class_name : str
            Class name for the title.
        confidence : float
            Prediction confidence for the title.
        save_path : str | Path, optional
            Path to save the figure.
        """
        # Load original image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot load image: {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

        # Create coloured heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title("Original", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=13)
        axes[2].axis("off")

        title = f"Predicted: {class_name}" if class_name else ""
        if confidence > 0:
            title += f" ({confidence:.1%})"
        fig.suptitle(title, fontsize=15, fontweight="bold")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


# =============================================================================
# Target Layer Resolver
# =============================================================================
def get_target_layer(model: nn.Module, model_type: str) -> str:
    """Get the correct target layer name for Grad-CAM per model type.

    Parameters
    ----------
    model : nn.Module
        The model.
    model_type : str
        Model type: 'efficientnet', 'resnet'.

    Returns
    -------
    str
        Target layer name.

    Raises
    ------
    ValueError
        If model_type is unknown.
    """
    layer_map = {
        "efficientnet": "model.features.8",
        "resnet": "model.layer4",
    }

    if model_type.lower() not in layer_map:
        available = list(layer_map.keys())
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {available}. "
            f"Or inspect model with: [n for n, _ in model.named_modules()]"
        )

    return layer_map[model_type.lower()]


# =============================================================================
# Batch Grad-CAM Generation
# =============================================================================
def batch_gradcam(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    class_names: List[str],
    save_dir: str | Path,
    target_layer_name: str,
    n_samples: int = 5,
) -> List[Tuple[str, str, np.ndarray]]:
    """Generate Grad-CAM for n_samples random images per class.

    Parameters
    ----------
    model : nn.Module
        Classification model.
    dataset : Dataset
        Image dataset with __getitem__ returning (tensor, label).
    class_names : list[str]
        Class names.
    save_dir : str | Path
        Directory to save visualisations.
    target_layer_name : str
        Target layer for Grad-CAM.
    n_samples : int
        Samples per class. Default 5.

    Returns
    -------
    list[tuple[str, str, np.ndarray]]
        List of (image_path, predicted_class, heatmap).
    """
    save_dir = Path(save_dir)
    device = next(model.parameters()).device

    gradcam = GradCAM(model, target_layer_name)
    results = []

    # Group indices by class
    if hasattr(dataset, "df"):
        df = dataset.df
    else:
        logger.warning("Dataset has no 'df' attribute. Sampling randomly.")
        return results

    for class_id, class_name in enumerate(class_names):
        class_indices = df[df["label_id"] == class_id].index.tolist()
        if not class_indices:
            continue

        np.random.shuffle(class_indices)
        selected = class_indices[:n_samples]

        class_save_dir = save_dir / "gradcam" / class_name
        class_save_dir.mkdir(parents=True, exist_ok=True)

        for idx in selected:
            img_tensor, label = dataset[idx]
            img_path = df.iloc[idx]["image_path"]

            heatmap = gradcam.generate(img_tensor, class_idx=class_id)

            # Get prediction
            with torch.no_grad():
                logits = model(img_tensor.unsqueeze(0).to(device))
                proba = torch.softmax(logits, dim=1)
                pred_class = proba.argmax(dim=1).item()
                confidence = proba[0, pred_class].item()

            save_path = class_save_dir / f"gradcam_{Path(img_path).stem}.png"
            gradcam.visualise(
                img_path, heatmap,
                class_name=class_names[pred_class],
                confidence=confidence,
                save_path=save_path,
            )

            results.append((img_path, class_names[pred_class], heatmap))

    gradcam.remove_hooks()
    logger.info(f"  Generated {len(results)} Grad-CAM visualisations")
    return results

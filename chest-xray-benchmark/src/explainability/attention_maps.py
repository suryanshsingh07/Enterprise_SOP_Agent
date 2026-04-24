"""
Attention Map Extraction — Chest X-ray Benchmark
==================================================
Extract and visualise attention maps from Swin Transformer.

Classes:
    SwinAttentionExtractor — hook-based attention weight extraction
Functions:
    compare_explanations — side-by-side CNN Grad-CAM vs Swin attention
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

logger = logging.getLogger(__name__)


# =============================================================================
# SwinAttentionExtractor
# =============================================================================
class SwinAttentionExtractor:
    """Extract attention maps from Swin Transformer via forward hooks.

    Registers hooks on all WindowAttention (or attention-like) modules
    to capture attention weight matrices.

    Parameters
    ----------
    model : nn.Module
        Swin Transformer model.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()
        self._attention_weights: List[torch.Tensor] = []
        self._hooks: List = []

        # Register hooks on all attention modules
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and (
                hasattr(module, "qkv") or "attention" in type(module).__name__.lower()
            ):
                hook = module.register_forward_hook(self._hook_fn)
                self._hooks.append(hook)

    def _hook_fn(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook to capture attention-related output."""
        if isinstance(output, torch.Tensor):
            self._attention_weights.append(output.detach().cpu())
        elif isinstance(output, tuple) and len(output) > 0:
            self._attention_weights.append(output[0].detach().cpu())

    def extract(self, img_tensor: torch.Tensor) -> List[np.ndarray]:
        """Extract attention maps from all hooked layers.

        Parameters
        ----------
        img_tensor : torch.Tensor
            Input image, shape (1, 3, 224, 224) or (3, 224, 224).

        Returns
        -------
        list[np.ndarray]
            List of attention maps, each normalised to [0, 1].
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device)

        self._attention_weights = []

        with torch.no_grad():
            _ = self.model(img_tensor)

        processed_maps = []
        for i, attn in enumerate(self._attention_weights):
            # Attempt to reshape to spatial grid
            if attn.dim() >= 2:
                # Average across all non-spatial dims
                while attn.dim() > 2:
                    attn = attn.mean(dim=1)

                attn_np = attn.numpy()

                # Try to reshape to square
                n_tokens = attn_np.shape[-1]
                side = int(np.sqrt(n_tokens))
                if side * side == n_tokens:
                    attn_map = attn_np.mean(axis=0).reshape(side, side)
                else:
                    attn_map = attn_np.mean(axis=0)

                # Normalise
                a_min, a_max = attn_map.min(), attn_map.max()
                if a_max - a_min > 1e-8:
                    attn_map = (attn_map - a_min) / (a_max - a_min)
                else:
                    attn_map = np.zeros_like(attn_map)

                processed_maps.append(attn_map)

        return processed_maps

    def visualise_stage(
        self,
        img_path: str | Path,
        attention_map: np.ndarray,
        stage_idx: int = 0,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Overlay attention map on the input image.

        Parameters
        ----------
        img_path : str | Path
            Original image path.
        attention_map : np.ndarray
            Attention map (H, W).
        stage_idx : int
            Stage index for title.
        save_path : str | Path, optional
            Save path.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot load: {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize attention map to image size
        attn_resized = cv2.resize(
            attention_map.astype(np.float32),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Apply colourmap
        attn_colored = cv2.applyColorMap(
            (attn_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
        )
        attn_colored = cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = cv2.addWeighted(img, 0.6, attn_colored, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title("Original", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(attn_resized, cmap="viridis")
        axes[1].set_title(f"Attention Map (Stage {stage_idx})", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=13)
        axes[2].axis("off")

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
# Side-by-Side Comparison
# =============================================================================
def compare_explanations(
    img_path: str | Path,
    cnn_model: nn.Module,
    vit_model: nn.Module,
    class_names: List[str],
    cnn_target_layer: str,
    save_path: Optional[str | Path] = None,
    img_size: int = 224,
) -> None:
    """Side-by-side comparison: Original | CNN Grad-CAM | Swin Attention.

    Parameters
    ----------
    img_path : str | Path
        Path to the image.
    cnn_model : nn.Module
        CNN model for Grad-CAM.
    vit_model : nn.Module
        Swin/ViT model for attention maps.
    class_names : list[str]
        Class names.
    cnn_target_layer : str
        Target layer for Grad-CAM.
    save_path : str | Path, optional
        Save path.
    img_size : int
        Image input size.
    """
    from src.explainability.gradcam import GradCAM
    from src.data.preprocessing import ChestXrayPreprocessor

    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Cannot load: {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_rgb, (img_size, img_size))

    # Prepare tensor
    preprocessor = ChestXrayPreprocessor(img_size=img_size)
    img_float = preprocessor.preprocess_rgb(str(img_path))
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float()

    device = next(cnn_model.parameters()).device

    # ── CNN Grad-CAM ──────────────────────────────────────────────────────────
    gradcam = GradCAM(cnn_model, cnn_target_layer)
    cnn_heatmap = gradcam.generate(img_tensor)
    gradcam.remove_hooks()

    cnn_colored = cv2.applyColorMap(
        (cnn_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cnn_colored = cv2.cvtColor(cnn_colored, cv2.COLOR_BGR2RGB)
    cnn_overlay = cv2.addWeighted(img_display, 0.6, cnn_colored, 0.4, 0)

    # ── Swin Attention ────────────────────────────────────────────────────────
    extractor = SwinAttentionExtractor(vit_model)
    attn_maps = extractor.extract(img_tensor)
    extractor.remove_hooks()

    if attn_maps:
        attn_map = attn_maps[-1]  # Use last stage
        attn_resized = cv2.resize(attn_map.astype(np.float32), (img_size, img_size))
        attn_colored = cv2.applyColorMap(
            (attn_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
        )
        attn_colored = cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB)
        attn_overlay = cv2.addWeighted(img_display, 0.6, attn_colored, 0.4, 0)
    else:
        attn_overlay = img_display

    # ── Get predictions ───────────────────────────────────────────────────────
    with torch.no_grad():
        cnn_logits = cnn_model(img_tensor.to(device))
        cnn_pred = class_names[cnn_logits.argmax(1).item()]

        vit_logits = vit_model(img_tensor.to(device))
        vit_pred = class_names[vit_logits.argmax(1).item()]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(cnn_overlay)
    axes[1].set_title(f"CNN Grad-CAM\nPred: {cnn_pred}", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(attn_overlay)
    axes[2].set_title(f"Swin Attention\nPred: {vit_pred}", fontsize=13)
    axes[2].axis("off")

    fig.suptitle("Explainability Comparison: CNN vs Transformer", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

"""
Detection Models — Chest X-ray Benchmark
==========================================
YOLO, Faster R-CNN, and Mask R-CNN wrappers for chest X-ray
bounding-box detection.

Classes:
    YOLODetector — YOLOv8 wrapper using ultralytics
    FasterRCNNChest — Faster R-CNN with ResNet-50-FPN backbone
    MaskRCNNChest — Mask R-CNN with ResNet-50-FPN backbone
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# YOLOv8 Wrapper
# =============================================================================
class YOLODetector:
    """YOLOv8 wrapper for chest X-ray detection.

    Uses the ultralytics library for training and inference.

    Parameters
    ----------
    model_name : str
        YOLO model variant. Default 'yolov8m.pt' (medium).
    n_classes : int
        Number of detection classes. Default 6.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        n_classes: int = 6,
    ) -> None:
        self.model_name = model_name
        self.n_classes = n_classes
        self._model = None

    def load(self) -> None:
        """Load the YOLO model."""
        from ultralytics import YOLO
        self._model = YOLO(self.model_name)
        logger.info(f"  Loaded YOLO model: {self.model_name}")

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 224,
        batch: int = 16,
        project: str = "outputs/yolo",
        name: str = "chest_xray",
        **kwargs: Any,
    ) -> Any:
        """Train the YOLO model.

        Parameters
        ----------
        data_yaml : str
            Path to YOLO dataset config YAML.
        epochs : int
            Number of training epochs.
        imgsz : int
            Input image size.
        batch : int
            Batch size.
        project : str
            Output project directory.
        name : str
            Run name.

        Returns
        -------
        Any
            Training results from ultralytics.
        """
        if self._model is None:
            self.load()

        results = self._model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            verbose=True,
            **kwargs,
        )
        return results

    def predict(
        self,
        source: str,
        conf: float = 0.25,
        save: bool = True,
    ) -> Any:
        """Run inference on images.

        Parameters
        ----------
        source : str
            Path to image or directory.
        conf : float
            Confidence threshold.
        save : bool
            Save annotated images.

        Returns
        -------
        Any
            Prediction results.
        """
        if self._model is None:
            self.load()
        return self._model.predict(source=source, conf=conf, save=save)

    def export(self, format: str = "onnx") -> str:
        """Export model to specified format.

        Parameters
        ----------
        format : str
            Export format: 'onnx', 'torchscript', etc.

        Returns
        -------
        str
            Path to exported model.
        """
        if self._model is None:
            self.load()
        return self._model.export(format=format)


# =============================================================================
# Faster R-CNN
# =============================================================================
class FasterRCNNChest(nn.Module):
    """Faster R-CNN with ResNet-50-FPN backbone for chest X-ray detection.

    Parameters
    ----------
    n_classes : int
        Number of classes (including background). Default 7 (6 + bg).
    pretrained_backbone : bool
        Use pretrained ResNet-50 backbone. Default True.
    min_size : int
        Minimum image size for the transform. Default 224.
    """

    def __init__(
        self,
        n_classes: int = 7,
        pretrained_backbone: bool = True,
        min_size: int = 224,
    ) -> None:
        super().__init__()
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
        self.model = fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=min_size,
        )

        # Replace the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Any:
        """Forward pass.

        Parameters
        ----------
        images : list[torch.Tensor]
            List of images, each (3, H, W).
        targets : list[dict], optional
            List of target dicts with 'boxes' and 'labels' keys.
            Required during training.

        Returns
        -------
        During training: dict of losses.
        During eval: list of prediction dicts.
        """
        return self.model(images, targets)


# =============================================================================
# Mask R-CNN
# =============================================================================
class MaskRCNNChest(nn.Module):
    """Mask R-CNN with ResNet-50-FPN for instance segmentation on CXR.

    Parameters
    ----------
    n_classes : int
        Number of classes (including background). Default 7.
    pretrained_backbone : bool
        Use pretrained backbone. Default True.
    min_size : int
        Minimum image size. Default 224.
    """

    def __init__(
        self,
        n_classes: int = 7,
        pretrained_backbone: bool = True,
        min_size: int = 224,
    ) -> None:
        super().__init__()
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn,
            MaskRCNN_ResNet50_FPN_Weights,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
        self.model = maskrcnn_resnet50_fpn(
            weights=weights,
            min_size=min_size,
        )

        # Replace box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, n_classes,
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Any:
        """Forward pass."""
        return self.model(images, targets)


if __name__ == "__main__":
    print("=" * 60)
    print("  Detection Models — Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Faster R-CNN
    frcnn = FasterRCNNChest(n_classes=7).to(device)
    dummy_imgs = [torch.randn(3, 224, 224).to(device)]
    frcnn.eval()
    with torch.no_grad():
        preds = frcnn(dummy_imgs)
    print(f"\n  Faster R-CNN: {len(preds)} predictions")
    for k, v in preds[0].items():
        print(f"    {k}: {v.shape if hasattr(v, 'shape') else v}")

    # Mask R-CNN
    mrcnn = MaskRCNNChest(n_classes=7).to(device)
    mrcnn.eval()
    with torch.no_grad():
        preds = mrcnn(dummy_imgs)
    print(f"\n  Mask R-CNN: {len(preds)} predictions")
    for k, v in preds[0].items():
        print(f"    {k}: {v.shape if hasattr(v, 'shape') else v}")

    # YOLO
    print(f"\n  YOLOv8: use YOLODetector.train() with dataset YAML")

    print("\n  ✓ Detection models demo complete.")

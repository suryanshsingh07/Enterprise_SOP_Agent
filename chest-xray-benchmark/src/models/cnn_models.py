"""
CNN Models — Chest X-ray Benchmark
====================================
EfficientNet-B3, ResNet-50, and PatchLSTM (RNN variant) for
multi-class chest X-ray classification with transfer learning.

Classes:
    EfficientNetChest — EfficientNet-B3 with custom head
    ResNet50Chest — ResNet-50 with custom head
    PatchLSTMChest — Patch-based BiLSTM using EfficientNet-B0 backbone
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    EfficientNet_B3_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EfficientNet-B3
# =============================================================================
class EfficientNetChest(nn.Module):
    """EfficientNet-B3 for 6-class chest X-ray classification.

    Architecture:
    - Backbone: EfficientNet-B3 (pretrained on ImageNet)
    - Custom head: Linear(1536→512) → ReLU → Dropout → Linear(512→6)
    - Supports freezing/unfreezing backbone blocks for 2-phase training

    Parameters
    ----------
    n_classes : int
        Number of output classes. Default 6.
    freeze_backbone : bool
        Freeze all backbone parameters on init. Default True.
    dropout : float
        Dropout rate in classifier head. Default 0.4.
    """

    def __init__(
        self,
        n_classes: int = 6,
        freeze_backbone: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        # Load pretrained EfficientNet-B3
        self.model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.model.classifier[1].in_features  # 1536 for B3
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, n_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 3, 224, 224).
        return_features : bool
            If True, also return feature maps before the classifier.

        Returns
        -------
        torch.Tensor
            Logits, shape (B, n_classes).
        torch.Tensor (optional)
            Features before classifier, shape (B, 1536).
        """
        features = self.model.features(x)
        pooled = self.model.avgpool(features)
        pooled = torch.flatten(pooled, 1)

        logits = self.model.classifier(pooled)

        if return_features:
            return logits, pooled
        return logits

    def unfreeze_top_blocks(self, n: int = 3) -> None:
        """Unfreeze the last n feature blocks for fine-tuning.

        Parameters
        ----------
        n : int
            Number of top blocks to unfreeze. Default 3.
        """
        total_blocks = len(self.model.features)
        for i in range(max(0, total_blocks - n), total_blocks):
            for param in self.model.features[i].parameters():
                param.requires_grad = True
        logger.info(f"  Unfroze top {n} blocks ({total_blocks - n}..{total_blocks - 1})")

    def get_target_layer(self) -> nn.Module:
        """Return the last convolutional layer for Grad-CAM.

        Returns
        -------
        nn.Module
            Last feature block (features[-1]).
        """
        return self.model.features[-1]


# =============================================================================
# ResNet-50
# =============================================================================
class ResNet50Chest(nn.Module):
    """ResNet-50 for 6-class chest X-ray classification.

    Architecture:
    - Backbone: ResNet-50 (pretrained on ImageNet)
    - Custom head: Linear(2048→512) → ReLU → Dropout(0.4) → Linear(512→6)

    Parameters
    ----------
    n_classes : int
        Number of output classes. Default 6.
    freeze_backbone : bool
        Freeze all backbone layers on init. Default True.
    dropout : float
        Dropout rate. Default 0.4.
    """

    def __init__(
        self,
        n_classes: int = 6,
        freeze_backbone: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        # Load pretrained ResNet-50
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze backbone
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        # Replace FC head
        in_features = self.model.fc.in_features  # 2048
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, n_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 3, 224, 224).
        return_features : bool
            If True, also return pooled features.

        Returns
        -------
        torch.Tensor
            Logits, shape (B, n_classes).
        """
        # Extract features up to avgpool
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        features = self.model.avgpool(x)
        features = torch.flatten(features, 1)

        logits = self.model.fc(features)

        if return_features:
            return logits, features
        return logits

    def unfreeze_top_blocks(self, n: int = 3) -> None:
        """Unfreeze the last n residual layers.

        ResNet-50 has 4 layers: layer1, layer2, layer3, layer4.

        Parameters
        ----------
        n : int
            Number of layers to unfreeze from the top. Default 3.
        """
        layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"  Unfroze top {n} residual layers")

    def get_target_layer(self) -> nn.Module:
        """Return the last residual layer for Grad-CAM."""
        return self.model.layer4


# =============================================================================
# PatchLSTM — RNN Variant for CXR
# =============================================================================
class PatchLSTMChest(nn.Module):
    """Patch-based BiLSTM for chest X-ray classification.

    Architecture:
    1. Split 224×224 image into n_patches horizontal strips
    2. Resize each strip to 64×64
    3. Extract features from each patch with frozen EfficientNet-B0
    4. Feed patch sequence to BiLSTM (hidden=256, layers=2)
    5. Classify from final hidden state: Linear(512→6)

    Parameters
    ----------
    n_classes : int
        Number of output classes. Default 6.
    n_patches : int
        Number of horizontal strips. Default 7.
    hidden_size : int
        LSTM hidden size. Default 256.
    num_layers : int
        Number of LSTM layers. Default 2.
    dropout : float
        LSTM + classifier dropout. Default 0.3.
    """

    def __init__(
        self,
        n_classes: int = 6,
        n_patches: int = 7,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.hidden_size = hidden_size
        self.patch_size = 64

        # Frozen EfficientNet-B0 backbone for patch feature extraction
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Determine feature dimension from backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        # EfficientNet-B0 outputs 1280 channels
        feature_dim = 1280

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classifier from BiLSTM output (2 * hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Logits, shape (B, n_classes).
        """
        B, C, H, W = x.shape
        patch_h = H // self.n_patches

        # Split into horizontal patches and extract features
        patch_features = []
        for i in range(self.n_patches):
            y_start = i * patch_h
            y_end = y_start + patch_h
            patch = x[:, :, y_start:y_end, :]  # (B, C, patch_h, W)

            # Resize patch to 64×64
            patch = F.interpolate(patch, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)

            # Extract features with frozen backbone
            with torch.no_grad():
                feat = self.feature_extractor(patch)  # (B, 1280, h, w)
            feat = self.pool(feat).flatten(1)  # (B, 1280)
            patch_features.append(feat)

        # Stack to sequence: (B, n_patches, 1280)
        sequence = torch.stack(patch_features, dim=1)

        # BiLSTM
        lstm_out, (h_n, _) = self.lstm(sequence)

        # Use last timestep output (concat forward + backward)
        last_output = lstm_out[:, -1, :]  # (B, 2*hidden_size)

        logits = self.classifier(last_output)
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("  CNN Models — Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 3, 224, 224).to(device)

    # EfficientNet-B3
    enet = EfficientNetChest(n_classes=6, freeze_backbone=True).to(device)
    out = enet(dummy)
    print(f"\n  EfficientNet-B3: input={dummy.shape} → output={out.shape}")
    enet.unfreeze_top_blocks(3)

    # ResNet-50
    rnet = ResNet50Chest(n_classes=6, freeze_backbone=True).to(device)
    out = rnet(dummy)
    print(f"  ResNet-50:       input={dummy.shape} → output={out.shape}")
    rnet.unfreeze_top_blocks(2)

    # PatchLSTM
    plstm = PatchLSTMChest(n_classes=6, n_patches=7).to(device)
    out = plstm(dummy)
    print(f"  PatchLSTM:       input={dummy.shape} → output={out.shape}")

    # Parameter counts
    for name, model in [("EfficientNet-B3", enet), ("ResNet-50", rnet), ("PatchLSTM", plstm)]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:20s}: total={total:>12,}  trainable={trainable:>12,}")

    print("\n  ✓ CNN models demo complete.")

"""
Training Losses — Chest X-ray Benchmark
=========================================
Focal Loss and other loss functions for handling class imbalance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification with class imbalance.

    Focal Loss reduces the relative loss for well-classified examples,
    focusing training on hard negatives. Critical for imbalanced CXR data
    where Normal class dominates.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : torch.Tensor, optional
        Per-class weight tensor of shape (C,). If None, uniform weighting.
        Typically set to inverse class frequencies.
    gamma : float
        Focusing parameter. Higher → more focus on hard examples.
        Default 2.0. Range: [0.5, 5.0].
    reduction : str
        Loss reduction: 'mean', 'sum', or 'none'. Default 'mean'.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw logits, shape (B, C).
        targets : torch.Tensor
            Ground truth class indices, shape (B,).

        Returns
        -------
        torch.Tensor
            Scalar loss (if reduction='mean' or 'sum'), otherwise (B,).
        """
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C)

        # Gather the probabilities of the target classes
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()  # (B, C)
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)

        # Compute cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)  # (B,)

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)
            alpha_t = alpha_t.gather(0, targets)  # (B,)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Parameters
    ----------
    smoothing : float
        Label smoothing factor. Default 0.1.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw logits, shape (B, C).
        targets : torch.Tensor
            Class indices, shape (B,).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        n_classes = inputs.shape[1]
        log_probs = F.log_softmax(inputs, dim=1)

        # Smooth target distribution
        targets_one_hot = F.one_hot(targets, num_classes=n_classes).float()
        smooth_targets = (
            (1.0 - self.smoothing) * targets_one_hot
            + self.smoothing / n_classes
        )

        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


def compute_class_weights(
    labels: torch.Tensor | list,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Formula: w_i = N / (n_classes * count_i)

    Parameters
    ----------
    labels : torch.Tensor | list
        All training labels.
    n_classes : int
        Number of classes.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Class weights, shape (n_classes,).
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    elif isinstance(labels, list):
        labels = __import__("numpy").array(labels)

    counts = __import__("numpy").bincount(labels.astype(int), minlength=n_classes)
    N = len(labels)

    # Avoid division by zero for missing classes
    weights = N / (n_classes * (counts + 1e-6))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    return weights

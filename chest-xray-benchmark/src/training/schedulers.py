"""
Learning Rate Schedulers — Chest X-ray Benchmark
===================================================
Wrappers and utilities for learning rate scheduling.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
    _LRScheduler,
)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "onecycle",
    epochs: int = 50,
    steps_per_epoch: int = 100,
    max_lr: float = 1e-3,
    pct_start: float = 0.1,
    step_size: int = 10,
    gamma: float = 0.5,
) -> _LRScheduler:
    """Build a learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer.
    scheduler_type : str
        Type: 'onecycle', 'cosine', 'step'. Default 'onecycle'.
    epochs : int
        Total training epochs.
    steps_per_epoch : int
        Steps per epoch (for OneCycleLR).
    max_lr : float
        Maximum learning rate (for OneCycleLR).
    pct_start : float
        Warmup fraction (for OneCycleLR). Default 0.1.
    step_size : int
        Step size for StepLR. Default 10.
    gamma : float
        Decay factor for StepLR/Cosine. Default 0.5.

    Returns
    -------
    _LRScheduler
        Configured scheduler.
    """
    if scheduler_type == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy="cos",
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=max_lr * 0.01,
        )
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer.
    warmup_epochs : int
        Number of warmup epochs.
    total_epochs : int
        Total training epochs.
    min_lr : float
        Minimum learning rate. Default 1e-6.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            import math
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]

"""
Trainer — Chest X-ray Benchmark
=================================
Main training loop with Focal Loss, AdamW + OneCycleLR, mixed precision,
and early stopping.

Classes:
    Trainer — complete training loop for any DL model
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.training.losses import FocalLoss, compute_class_weights

logger = logging.getLogger(__name__)


class Trainer:
    """Complete training loop for deep learning models.

    Features:
    - Focal Loss with per-class alpha weights (auto-computed)
    - AdamW optimizer with OneCycleLR (10% warmup, cosine decay)
    - Mixed precision training (torch.cuda.amp)
    - Early stopping on validation macro F1
    - Best checkpoint saving

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    n_classes : int
        Number of output classes. Default 6.
    device : torch.device | str
        Compute device. Default 'cuda'.
    epochs : int
        Number of training epochs. Default 50.
    lr : float
        Maximum learning rate. Default 1e-3.
    weight_decay : float
        AdamW weight decay. Default 1e-4.
    use_focal_loss : bool
        If True, use Focal Loss; otherwise CrossEntropy. Default True.
    focal_gamma : float
        Focal Loss gamma parameter. Default 2.0.
    patience : int
        Early stopping patience (epochs). Default 10.
    use_amp : bool
        Use automatic mixed precision. Default True.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_classes: int = 6,
        device: torch.device | str = "cuda",
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        patience: int = 10,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_classes = n_classes
        self.device = torch.device(device) if isinstance(device, str) else device
        self.epochs = epochs
        self.patience = patience
        self.use_amp = use_amp and self.device.type == "cuda"

        self.model.to(self.device)

        # Compute class weights from training labels
        all_labels = self._collect_labels(train_loader)
        class_weights = compute_class_weights(all_labels, n_classes, self.device)
        logger.info(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler: OneCycleLR with 10% warmup
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,       # 10% warmup
            anneal_strategy="cos",
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_f1": [], "lr": []}

    @staticmethod
    def _collect_labels(loader: DataLoader) -> np.ndarray:
        """Collect all labels from a DataLoader."""
        labels = []
        for _, y in loader:
            labels.extend(y.numpy().tolist())
        return np.array(labels, dtype=np.int64)

    def train_epoch(self) -> float:
        """Run one training epoch.

        Returns
        -------
        float
            Average training loss.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and compute macro F1 score.

        Returns
        -------
        float
            Macro F1 score on validation set.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return macro_f1

    def fit(self, save_path: str | Path = "outputs/checkpoints/best_model.pth") -> dict:
        """Full training loop with early stopping and checkpoint saving.

        Parameters
        ----------
        save_path : str | Path
            Path to save the best model checkpoint.

        Returns
        -------
        dict
            Training history with train_loss, val_f1, and lr per epoch.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n  Training: {self.epochs} epochs, device={self.device}")
        print(f"  Patience: {self.patience}, AMP: {self.use_amp}")
        print(f"  {'Epoch':>5s} │ {'Loss':>8s} │ {'Val F1':>8s} │ {'LR':>10s} │ {'Status'}")
        print(f"  {'─' * 55}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_f1 = self.validate()

            # Current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            # Check for improvement
            status = ""
            if val_f1 > self.best_f1 + 0.001:
                self.best_f1 = val_f1
                self.best_epoch = epoch
                self.patience_counter = 0
                status = "★ best"

                # Save checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_f1": self.best_f1,
                    "n_classes": self.n_classes,
                }, save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    status = "⛔ early stop"

            elapsed = time.time() - t0
            print(
                f"  {epoch:5d} │ {train_loss:8.4f} │ {val_f1:8.4f} │ "
                f"{current_lr:10.2e} │ {status}  ({elapsed:.1f}s)"
            )

            if self.patience_counter >= self.patience:
                print(f"\n  Early stopping at epoch {epoch}. Best: epoch {self.best_epoch} (F1={self.best_f1:.4f})")
                break

        print(f"\n  Training complete. Best F1: {self.best_f1:.4f} (epoch {self.best_epoch})")
        print(f"  Checkpoint: {save_path}")

        return self.history

    def load_best(self, path: str | Path) -> None:
        """Load the best checkpoint back into the model.

        Parameters
        ----------
        path : str | Path
            Path to the saved checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"  Loaded best checkpoint (epoch {checkpoint['epoch']}, F1={checkpoint['best_f1']:.4f})")

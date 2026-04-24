"""
Training Callbacks — Chest X-ray Benchmark
============================================
EarlyStopping, ModelCheckpoint, and TrainingLogger for experiment tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# EarlyStopping
# =============================================================================
class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping. Default 10.
    mode : str
        'max' for metrics to maximise (F1, acc), 'min' for loss. Default 'max'.
    min_delta : float
        Minimum change to qualify as improvement. Default 0.001.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.001,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        metric : float
            Current metric value.

        Returns
        -------
        bool
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = metric
            return False

        if self.mode == "max":
            improved = metric > self.best_value + self.min_delta
        else:
            improved = metric < self.best_value - self.min_delta

        if improved:
            self.best_value = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


# =============================================================================
# ModelCheckpoint
# =============================================================================
class ModelCheckpoint:
    """Save the best model checkpoint when a metric improves.

    Parameters
    ----------
    save_path : str | Path
        Path to save the checkpoint.
    mode : str
        'max' or 'min'. Default 'max'.
    """

    def __init__(
        self,
        save_path: str | Path,
        mode: str = "max",
    ) -> None:
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0

    def __call__(
        self,
        model: nn.Module,
        metric: float,
        epoch: int,
        config: Optional[dict] = None,
    ) -> bool:
        """Save checkpoint if metric improved.

        Parameters
        ----------
        model : nn.Module
            Model to save.
        metric : float
            Current metric value.
        epoch : int
            Current epoch.
        config : dict, optional
            Training configuration to save as metadata.

        Returns
        -------
        bool
            True if checkpoint was saved.
        """
        if self.best_value is None:
            improved = True
        elif self.mode == "max":
            improved = metric > self.best_value
        else:
            improved = metric < self.best_value

        if improved:
            self.best_value = metric
            self.best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metric_value": metric,
            }
            if config:
                checkpoint["config"] = config

            torch.save(checkpoint, self.save_path)

            # Save metadata as JSON
            meta_path = self.save_path.with_suffix(".json")
            meta = {
                "epoch": epoch,
                "metric_value": float(metric),
                "checkpoint_path": str(self.save_path),
            }
            if config:
                meta["config"] = config
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"  Checkpoint saved: epoch={epoch}, metric={metric:.4f}")
            return True

        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best checkpoint into the model.

        Parameters
        ----------
        model : nn.Module
            Model to load weights into.

        Returns
        -------
        nn.Module
            Model with loaded weights.
        """
        if not self.save_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.save_path}")

        checkpoint = torch.load(self.save_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"  Loaded best checkpoint: epoch={checkpoint['epoch']}, "
            f"metric={checkpoint['metric_value']:.4f}"
        )
        return model


# =============================================================================
# TrainingLogger
# =============================================================================
class TrainingLogger:
    """Log training metrics to CSV and generate plots.

    Parameters
    ----------
    log_dir : str | Path
        Directory to save log files and plots.
    experiment_name : str
        Name of the experiment. Default 'experiment'.
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str = "experiment",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.csv_path = self.log_dir / f"{experiment_name}_log.csv"

        self.records: list[dict] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_f1: float,
        lr: float,
        **kwargs: float,
    ) -> None:
        """Log metrics for one epoch.

        Parameters
        ----------
        epoch : int
            Epoch number.
        train_loss : float
            Training loss.
        val_f1 : float
            Validation macro F1 score.
        lr : float
            Current learning rate.
        **kwargs : float
            Additional metrics.
        """
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_f1": round(val_f1, 4),
            "lr": lr,
            **{k: round(v, 4) for k, v in kwargs.items()},
        }
        self.records.append(record)

        # Append to CSV
        df = pd.DataFrame([record])
        df.to_csv(
            self.csv_path,
            mode="a",
            header=not self.csv_path.exists() or len(self.records) == 1,
            index=False,
        )

    def plot_curves(self, save_path: Optional[str | Path] = None) -> None:
        """Plot training loss and validation F1 curves.

        Parameters
        ----------
        save_path : str | Path, optional
            Path to save the plot. If None, saves to log_dir.
        """
        if not self.records:
            logger.warning("No records to plot.")
            return

        df = pd.DataFrame(self.records)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax1.plot(df["epoch"], df["train_loss"], "b-", linewidth=2, label="Train Loss")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title(f"{self.experiment_name} — Training Loss", fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # F1 curve
        ax2.plot(df["epoch"], df["val_f1"], "g-", linewidth=2, label="Val Macro F1")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Macro F1", fontsize=12)
        ax2.set_title(f"{self.experiment_name} — Validation F1", fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = save_path or self.log_dir / f"{self.experiment_name}_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Training curves saved: {save_path}")

    def summary(self) -> None:
        """Print training summary."""
        if not self.records:
            print("  No training records found.")
            return

        df = pd.DataFrame(self.records)
        best_idx = df["val_f1"].idxmax()
        best = df.iloc[best_idx]

        print(f"\n  {'='*50}")
        print(f"  Training Summary: {self.experiment_name}")
        print(f"  {'='*50}")
        print(f"    Total epochs:    {len(df)}")
        print(f"    Best epoch:      {int(best['epoch'])}")
        print(f"    Best Val F1:     {best['val_f1']:.4f}")
        print(f"    Final loss:      {df.iloc[-1]['train_loss']:.4f}")
        print(f"    Log file:        {self.csv_path}")

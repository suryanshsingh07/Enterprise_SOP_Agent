#!/usr/bin/env python3
"""
04 — Train CNN Models
======================
Two-phase transfer learning for EfficientNet-B3 and ResNet-50.
Phase 1: Freeze backbone, train head (10 epochs)
Phase 2: Unfreeze top blocks, fine-tune (40 epochs)

Usage:
    python scripts/04_train_cnn.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import ChestXrayDataset
from src.models.cnn_models import EfficientNetChest, ResNet50Chest
from src.training.trainer import Trainer
from src.training.utils import set_seed, get_device, count_parameters, create_dataloaders


def train_cnn_model(
    model_class: type,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Train a CNN model with 2-phase transfer learning.

    Parameters
    ----------
    model_class : type
        Model class (EfficientNetChest or ResNet50Chest).
    model_name : str
        Name for logging and saving.
    train_df : pd.DataFrame
        Training manifest.
    val_df : pd.DataFrame
        Validation manifest.
    device : torch.device
        Compute device.
    save_dir : Path
        Checkpoint save directory.

    Returns
    -------
    dict
        Training results with best F1 per phase.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    results = {}

    # ── Phase 1: Frozen backbone ──────────────────────────────────────────────
    print(f"\n  Phase 1: Freeze backbone, train head (10 epochs)")
    model = model_class(n_classes=6, freeze_backbone=True, dropout=0.4)
    model.to(device)

    params = count_parameters(model)
    print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    train_dataset = ChestXrayDataset(train_df, split="train", img_size=224)
    val_dataset = ChestXrayDataset(val_df, split="val", img_size=224)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=6,
        device=device,
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        use_focal_loss=True,
        focal_gamma=2.0,
        patience=10,
    )

    history1 = trainer.fit(save_path=save_dir / f"{model_name}_phase1.pth")
    results["phase1_best_f1"] = trainer.best_f1

    # ── Phase 2: Unfreeze top blocks ──────────────────────────────────────────
    print(f"\n  Phase 2: Unfreeze top 3 blocks, fine-tune (40 epochs)")
    trainer.load_best(save_dir / f"{model_name}_phase1.pth")
    model.unfreeze_top_blocks(n=3)

    params = count_parameters(model)
    print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    trainer2 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=6,
        device=device,
        epochs=40,
        lr=1e-4,
        weight_decay=1e-4,
        use_focal_loss=True,
        focal_gamma=2.0,
        patience=10,
    )

    history2 = trainer2.fit(save_path=save_dir / f"{model_name}_best.pth")
    results["phase2_best_f1"] = trainer2.best_f1
    results["best_epoch"] = trainer2.best_epoch

    # Combine histories
    results["train_loss"] = history1["train_loss"] + history2["train_loss"]
    results["val_f1"] = history1["val_f1"] + history2["val_f1"]

    return results


def main() -> None:
    """Main CNN training pipeline."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — CNN TRAINING")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    manifests_dir = Path("data/manifests")
    save_dir = Path("outputs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load manifests
    for split in ["train", "val"]:
        if not (manifests_dir / f"{split}.csv").exists():
            print(f"\n  ⚠ Manifest not found: {manifests_dir / split}.csv")
            print("  → Run 01_build_dataset.py first.")
            return

    train_df = pd.read_csv(manifests_dir / "train.csv")
    val_df = pd.read_csv(manifests_dir / "val.csv")

    print(f"\n  Train: {len(train_df):,} images")
    print(f"  Val:   {len(val_df):,} images")

    all_results = {}

    # Train EfficientNet-B3
    all_results["EfficientNet-B3"] = train_cnn_model(
        EfficientNetChest, "efficientnet_b3", train_df, val_df, device, save_dir,
    )

    # Train ResNet-50
    all_results["ResNet-50"] = train_cnn_model(
        ResNet50Chest, "resnet50", train_df, val_df, device, save_dir,
    )

    # Save results
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serialisable format
    serialisable = {}
    for name, res in all_results.items():
        serialisable[name] = {
            "phase1_best_f1": res["phase1_best_f1"],
            "phase2_best_f1": res["phase2_best_f1"],
            "best_epoch": res["best_epoch"],
        }

    with open(experiments_dir / "cnn_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    print("\n" + "=" * 60)
    print("  CNN TRAINING — RESULTS")
    print("=" * 60)
    for name, res in serialisable.items():
        print(f"  {name:20s}: Phase1 F1={res['phase1_best_f1']:.4f}, "
              f"Phase2 F1={res['phase2_best_f1']:.4f}")
    print("\n  ✓ CNN training complete!")


if __name__ == "__main__":
    main()

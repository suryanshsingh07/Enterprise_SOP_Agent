#!/usr/bin/env python3
"""
05 — Train Transformer Models
===============================
Two-phase training for Swin-Tiny and ViT-B/16, plus autoencoder training.

Usage:
    python scripts/05_train_transformer.py
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
from src.models.transformer_models import (
    SwinChest,
    ViTChest,
    ChestAutoencoder,
    train_autoencoder,
    compute_anomaly_threshold,
)
from src.training.trainer import Trainer
from src.training.utils import set_seed, get_device, count_parameters


def train_transformer_model(
    model_class: type,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    save_dir: Path,
    unfreeze_method: str = "stages",
) -> dict:
    """Train a transformer model with 2-phase transfer learning.

    Parameters
    ----------
    model_class : type
        Model class (SwinChest or ViTChest).
    model_name : str
        Name for logging and saving.
    train_df, val_df : pd.DataFrame
        Data manifests.
    device : torch.device
        Compute device.
    save_dir : Path
        Checkpoint directory.
    unfreeze_method : str
        'stages' for Swin, 'blocks' for ViT.

    Returns
    -------
    dict
        Training results.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    results = {}

    # Phase 1: Frozen backbone
    print(f"\n  Phase 1: Frozen backbone, train head (10 epochs)")
    model = model_class(n_classes=6, freeze_backbone=True)
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
        model=model, train_loader=train_loader,
        val_loader=val_loader, n_classes=6, device=device,
        epochs=10, lr=1e-3, use_focal_loss=True, patience=10,
    )

    history1 = trainer.fit(save_path=save_dir / f"{model_name}_phase1.pth")
    results["phase1_best_f1"] = trainer.best_f1

    # Phase 2: Unfreeze
    print(f"\n  Phase 2: Unfreeze top layers, fine-tune (40 epochs)")
    trainer.load_best(save_dir / f"{model_name}_phase1.pth")

    if unfreeze_method == "stages":
        model.unfreeze_last_n_stages(n=2)
    else:
        model.unfreeze_last_n_blocks(n=4)

    params = count_parameters(model)
    print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    trainer2 = Trainer(
        model=model, train_loader=train_loader,
        val_loader=val_loader, n_classes=6, device=device,
        epochs=40, lr=1e-4, use_focal_loss=True, patience=10,
    )

    history2 = trainer2.fit(save_path=save_dir / f"{model_name}_best.pth")
    results["phase2_best_f1"] = trainer2.best_f1
    results["best_epoch"] = trainer2.best_epoch

    return results


def main() -> None:
    """Main transformer training pipeline."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — TRANSFORMER TRAINING")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    manifests_dir = Path("data/manifests")
    save_dir = Path("outputs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        if not (manifests_dir / f"{split}.csv").exists():
            print(f"\n  ⚠ Run 01_build_dataset.py first.")
            return

    train_df = pd.read_csv(manifests_dir / "train.csv")
    val_df = pd.read_csv(manifests_dir / "val.csv")

    all_results = {}

    # Train Swin-Tiny
    all_results["Swin-Tiny"] = train_transformer_model(
        SwinChest, "swin_tiny", train_df, val_df, device, save_dir,
        unfreeze_method="stages",
    )

    # Train ViT-B/16
    all_results["ViT-B/16"] = train_transformer_model(
        ViTChest, "vit_base16", train_df, val_df, device, save_dir,
        unfreeze_method="blocks",
    )

    # Train Autoencoder on Normal class
    print(f"\n{'='*60}")
    print(f"  Training: Convolutional Autoencoder (Normal only)")
    print(f"{'='*60}")

    normal_train = train_df[train_df["label_id"] == 0]
    normal_val = val_df[val_df["label_id"] == 0]
    print(f"  Normal train: {len(normal_train):,}, Normal val: {len(normal_val):,}")

    if len(normal_train) > 0:
        normal_train_ds = ChestXrayDataset(normal_train, split="train", img_size=224)
        normal_val_ds = ChestXrayDataset(normal_val, split="val", img_size=224)

        normal_train_loader = torch.utils.data.DataLoader(
            normal_train_ds, batch_size=32, shuffle=True, num_workers=4,
        )
        normal_val_loader = torch.utils.data.DataLoader(
            normal_val_ds, batch_size=32, shuffle=False, num_workers=4,
        )

        ae_model = ChestAutoencoder(latent_dim=256)
        ae_model, ae_losses = train_autoencoder(
            ae_model, normal_train_loader, epochs=50, lr=1e-3, device=device,
        )

        threshold = compute_anomaly_threshold(ae_model, normal_val_loader, device=device)
        all_results["Autoencoder"] = {
            "final_loss": ae_losses[-1],
            "threshold": threshold,
        }

        torch.save(ae_model.state_dict(), save_dir / "autoencoder_best.pth")

    # Save results
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)

    serialisable = {}
    for name, res in all_results.items():
        serialisable[name] = {k: v for k, v in res.items()
                               if isinstance(v, (int, float, str))}

    with open(experiments_dir / "transformer_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRANSFORMER TRAINING — RESULTS")
    print("=" * 60)
    for name, res in serialisable.items():
        print(f"  {name}: {res}")
    print("\n  ✓ Transformer training complete!")


if __name__ == "__main__":
    main()

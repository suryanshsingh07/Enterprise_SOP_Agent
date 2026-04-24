#!/usr/bin/env python3
"""
08 — Train Detection Models (Faster R-CNN / Mask R-CNN)
========================================================
Train Faster R-CNN and Mask R-CNN on chest X-ray detection.

Usage:
    python scripts/08_train_detection.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.detection_dataset import DetectionDataset, collate_fn
from src.models.detection_models import FasterRCNNChest, MaskRCNNChest
from src.training.utils import set_seed, get_device


def train_detection_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    device: torch.device,
    epochs: int = 20,
    lr: float = 5e-4,
    save_dir: Path = Path("outputs/checkpoints"),
) -> dict:
    """Train a detection model.

    Parameters
    ----------
    model : nn.Module
        Detection model (Faster/Mask R-CNN).
    train_loader, val_loader : DataLoader
        Data loaders.
    model_name : str
        Name for logging.
    device : torch.device
        Compute device.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    save_dir : Path
        Checkpoint save directory.

    Returns
    -------
    dict
        Training results.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float("inf")
    history = {"train_loss": []}

    print(f"\n  Training {model_name}: {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running_loss += losses.item()
            n_batches += 1

        lr_scheduler.step()
        avg_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / f"{model_name}_best.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}")

    print(f"  Best loss: {best_loss:.4f}")
    return {"best_loss": best_loss, "epochs": epochs}


def main() -> None:
    """Main detection training pipeline."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — DETECTION TRAINING")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    manifests_dir = Path("data/manifests")
    if not (manifests_dir / "train.csv").exists():
        print("\n  ⚠ Run 01_build_dataset.py first.")
        return

    train_df = pd.read_csv(manifests_dir / "train.csv")
    val_df = pd.read_csv(manifests_dir / "val.csv")

    train_ds = DetectionDataset(train_df, img_size=224)
    val_ds = DetectionDataset(val_df, img_size=224)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        num_workers=4, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )

    save_dir = Path("outputs/checkpoints")
    all_results = {}

    # Faster R-CNN
    print("\n  Training Faster R-CNN...")
    frcnn = FasterRCNNChest(n_classes=7)
    all_results["Faster R-CNN"] = train_detection_model(
        frcnn, train_loader, val_loader,
        "faster_rcnn", device, epochs=20, save_dir=save_dir,
    )

    # Mask R-CNN
    print("\n  Training Mask R-CNN...")
    mrcnn = MaskRCNNChest(n_classes=7)
    all_results["Mask R-CNN"] = train_detection_model(
        mrcnn, train_loader, val_loader,
        "mask_rcnn", device, epochs=20, save_dir=save_dir,
    )

    # Save results
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / "detection_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("  ✓ Detection training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

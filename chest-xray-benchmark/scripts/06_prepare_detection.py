#!/usr/bin/env python3
"""
06 — Prepare Detection Annotations
====================================
Convert classification manifests to COCO and YOLO detection formats.

Usage:
    python scripts/06_prepare_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation_converter import (
    create_coco_annotations,
    create_yolo_annotations,
)


def main() -> None:
    """Prepare detection annotations."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — DETECTION ANNOTATION PREP")
    print("=" * 60)

    manifests_dir = Path("data/manifests")
    annotations_dir = Path("data/annotations")

    for split in ["train", "val"]:
        csv_path = manifests_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"\n  ⚠ {csv_path} not found. Run 01_build_dataset.py first.")
            return

    train_df = pd.read_csv(manifests_dir / "train.csv")
    val_df = pd.read_csv(manifests_dir / "val.csv")

    # Create COCO annotations
    print("\n  Creating COCO annotations...")
    create_coco_annotations(
        train_df, "data/processed/train",
        annotations_dir / "coco" / "train.json",
    )
    create_coco_annotations(
        val_df, "data/processed/val",
        annotations_dir / "coco" / "val.json",
    )

    # Create YOLO annotations
    print("  Creating YOLO annotations...")
    create_yolo_annotations(
        train_df, "data/processed/train",
        annotations_dir / "yolo" / "train",
    )
    create_yolo_annotations(
        val_df, "data/processed/val",
        annotations_dir / "yolo" / "val",
    )

    print("\n" + "=" * 60)
    print("  ✓ Detection annotations prepared!")
    print("=" * 60)


if __name__ == "__main__":
    main()

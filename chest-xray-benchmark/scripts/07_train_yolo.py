#!/usr/bin/env python3
"""
07 — Train YOLOv8
==================
Train YOLOv8m on chest X-ray detection using ultralytics.

Usage:
    python scripts/07_train_yolo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detection_models import YOLODetector


def main() -> None:
    """Train YOLOv8 on chest X-ray data."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — YOLO TRAINING")
    print("=" * 60)

    data_yaml = "configs/yolo_dataset.yaml"
    if not Path(data_yaml).exists():
        print(f"\n  ⚠ {data_yaml} not found.")
        return

    detector = YOLODetector(model_name="yolov8m.pt", n_classes=6)

    print("\n  Starting YOLOv8m training...")
    results = detector.train(
        data_yaml=data_yaml,
        epochs=100,
        imgsz=224,
        batch=16,
        project="outputs/yolo",
        name="chest_xray_v1",
    )

    print("\n" + "=" * 60)
    print("  ✓ YOLO training complete!")
    print(f"  Results saved to: outputs/yolo/chest_xray_v1/")
    print("=" * 60)


if __name__ == "__main__":
    main()

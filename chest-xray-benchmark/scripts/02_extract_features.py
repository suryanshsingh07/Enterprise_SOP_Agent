#!/usr/bin/env python3
"""
02 — Extract Features
======================
Extract HOG + LBP features from processed images, fit PCA on training set.

Usage:
    python scripts/02_extract_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import ChestXrayPreprocessor
from src.features.classical_features import ClassicalFeatureExtractor


def main() -> None:
    """Main feature extraction pipeline."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — FEATURE EXTRACTION")
    print("=" * 60)

    output_dir = Path("outputs/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = Path("data/manifests")

    # Load manifests
    for split in ["train", "val", "test"]:
        csv_path = manifests_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"\n  ⚠ Manifest not found: {csv_path}")
            print("  → Run 01_build_dataset.py first.")
            return

    train_df = pd.read_csv(manifests_dir / "train.csv")
    val_df = pd.read_csv(manifests_dir / "val.csv")
    test_df = pd.read_csv(manifests_dir / "test.csv")

    print(f"\n  Train: {len(train_df):,} images")
    print(f"  Val:   {len(val_df):,} images")
    print(f"  Test:  {len(test_df):,} images")

    preprocessor = ChestXrayPreprocessor(img_size=224, clahe_clip=2.0)
    extractor = ClassicalFeatureExtractor(pca_dim=512)

    # Extract training features and fit PCA
    print("\n  [1/3] Extracting training features...")
    train_images = []
    train_labels = []
    for _, row in train_df.iterrows():
        try:
            img = preprocessor.preprocess_numpy(row["image_path"])
            train_images.append(img)
            train_labels.append(int(row["label_id"]))
        except Exception:
            pass

    X_train = extractor.fit_transform(train_images)
    y_train = np.array(train_labels, dtype=np.int64)

    # Validation features
    print("  [2/3] Extracting validation features...")
    val_images, val_labels = [], []
    for _, row in val_df.iterrows():
        try:
            img = preprocessor.preprocess_numpy(row["image_path"])
            val_images.append(img)
            val_labels.append(int(row["label_id"]))
        except Exception:
            pass

    X_val = extractor.transform(val_images)
    y_val = np.array(val_labels, dtype=np.int64)

    # Test features
    print("  [3/3] Extracting test features...")
    test_images, test_labels = [], []
    for _, row in test_df.iterrows():
        try:
            img = preprocessor.preprocess_numpy(row["image_path"])
            test_images.append(img)
            test_labels.append(int(row["label_id"]))
        except Exception:
            pass

    X_test = extractor.transform(test_images)
    y_test = np.array(test_labels, dtype=np.int64)

    # Save
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    extractor.save(output_dir / "feature_extractor.joblib")

    print(f"\n  Saved to {output_dir}:")
    print(f"    X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"    X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"    X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    print("\n" + "=" * 60)
    print("  ✓ Feature extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

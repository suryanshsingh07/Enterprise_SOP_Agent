"""
Feature Pipeline — Chest X-ray Benchmark
==========================================
End-to-end pipeline orchestrating feature extraction, PCA fitting,
and serialisation for classical ML experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.preprocessing import ChestXrayPreprocessor
from src.features.classical_features import (
    ClassicalFeatureExtractor,
    extract_features_from_dataframe,
)

logger = logging.getLogger(__name__)


def run_feature_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = "outputs/features",
    img_size: int = 224,
    pca_dim: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the complete feature extraction pipeline.

    1. Extract raw HOG+LBP features from all splits
    2. Fit StandardScaler + PCA on training data
    3. Transform all splits
    4. Save extractor and feature arrays

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with ``image_path`` and ``label_id``.
    val_df : pd.DataFrame
        Validation DataFrame.
    test_df : pd.DataFrame
        Test DataFrame.
    output_dir : str | Path
        Directory to save features and extractor.
    img_size : int
        Image resize target. Default 224.
    pca_dim : int
        PCA output dimensions. Default 512.

    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = ChestXrayPreprocessor(img_size=img_size)
    extractor = ClassicalFeatureExtractor(pca_dim=pca_dim)

    # Extract raw features from training set
    print("\n  [1/4] Extracting raw features from training set...")
    X_train_raw, y_train = extract_features_from_dataframe(
        train_df, extractor, preprocessor
    )

    # Fit scaler + PCA on training features
    print("\n  [2/4] Fitting StandardScaler + PCA...")
    X_train_images = []
    for _, row in train_df.iterrows():
        try:
            img = preprocessor.preprocess_numpy(row["image_path"])
            X_train_images.append(img)
        except Exception:
            pass
    X_train = extractor.fit_transform(X_train_images)

    # Extract and transform validation features
    print("\n  [3/4] Extracting validation features...")
    X_val_images = []
    y_val_list = []
    for _, row in val_df.iterrows():
        try:
            img = preprocessor.preprocess_numpy(row["image_path"])
            X_val_images.append(img)
            y_val_list.append(int(row["label_id"]))
        except Exception:
            pass
    X_val = extractor.transform(X_val_images)
    y_val = np.array(y_val_list, dtype=np.int64)

    # Extract and transform test features
    print("\n  [4/4] Extracting test features...")
    X_test_images = []
    y_test_list = []
    for _, row in test_df.iterrows():
        try:
            img = preprocessor.preprocess_numpy(row["image_path"])
            X_test_images.append(img)
            y_test_list.append(int(row["label_id"]))
        except Exception:
            pass
    X_test = extractor.transform(X_test_images)
    y_test = np.array(y_test_list, dtype=np.int64)

    y_train = np.array([int(row["label_id"]) for _, row in train_df.iterrows()
                        if Path(row["image_path"]).exists()], dtype=np.int64)[:len(X_train)]

    # Save everything
    extractor.save(output_dir / "feature_extractor.joblib")
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    print(f"\n  Features saved to {output_dir}")
    print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"    X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

"""
Classical Feature Extraction — Chest X-ray Benchmark
=====================================================
HOG + LBP feature extraction with PCA dimensionality reduction.

Classes:
    ClassicalFeatureExtractor — extract HOG(6084-d) + LBP(26-d), PCA → 512-d
Functions:
    extract_features_from_dataframe — batch extraction with progress bar
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# ClassicalFeatureExtractor
# =============================================================================
class ClassicalFeatureExtractor:
    """Extract classical features (HOG + LBP) and reduce with PCA.

    Feature pipeline:
    1. HOG: 9 orientations, 16×16 pixels/cell, 2×2 cells/block → 6084-d
    2. LBP: radius=3, 24 points, uniform → 26-d histogram
    3. Concatenate → 6110-d
    4. StandardScaler → PCA(512) → 512-d

    Parameters
    ----------
    pca_dim : int
        Target PCA dimensions. Default 512.
    hog_orientations : int
        Number of HOG orientation bins. Default 9.
    hog_pixels_per_cell : tuple[int, int]
        HOG cell size. Default (16, 16).
    hog_cells_per_block : tuple[int, int]
        HOG block size. Default (2, 2).
    lbp_radius : int
        LBP radius. Default 3.
    lbp_n_points : int
        LBP number of sampling points. Default 24.
    """

    def __init__(
        self,
        pca_dim: int = 512,
        hog_orientations: int = 9,
        hog_pixels_per_cell: Tuple[int, int] = (16, 16),
        hog_cells_per_block: Tuple[int, int] = (2, 2),
        lbp_radius: int = 3,
        lbp_n_points: int = 24,
    ) -> None:
        self.pca_dim = pca_dim
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self._is_fitted = False

    def extract_hog(self, img_gray: np.ndarray) -> np.ndarray:
        """Extract HOG features from a grayscale image.

        Parameters
        ----------
        img_gray : np.ndarray
            Grayscale image (H, W), dtype uint8 or float.

        Returns
        -------
        np.ndarray
            HOG feature vector. For 224×224 with default params: 6084-d.
        """
        features = hog(
            img_gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return features.astype(np.float32)

    def extract_lbp(self, img_gray: np.ndarray) -> np.ndarray:
        """Extract LBP histogram from a grayscale image.

        Parameters
        ----------
        img_gray : np.ndarray
            Grayscale image (H, W), dtype uint8 or float.

        Returns
        -------
        np.ndarray
            LBP histogram (n_points + 2 = 26 bins for default params).
        """
        lbp_image = local_binary_pattern(
            img_gray,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method="uniform",
        )
        # Compute histogram of uniform LBP patterns
        n_bins = self.lbp_n_points + 2  # uniform patterns + 1 non-uniform bin
        hist, _ = np.histogram(
            lbp_image.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True,
        )
        return hist.astype(np.float32)

    def extract_single(self, img_gray: np.ndarray) -> np.ndarray:
        """Extract concatenated HOG + LBP features from a single image.

        Parameters
        ----------
        img_gray : np.ndarray
            Grayscale image (H, W).

        Returns
        -------
        np.ndarray
            Concatenated feature vector (6084 + 26 = 6110-d).
        """
        hog_feat = self.extract_hog(img_gray)
        lbp_feat = self.extract_lbp(img_gray)
        return np.concatenate([hog_feat, lbp_feat])

    def fit_transform(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features, fit StandardScaler + PCA, and transform.

        This should only be called on the **training** set.

        Parameters
        ----------
        images : list[np.ndarray]
            List of grayscale images (each H, W).

        Returns
        -------
        np.ndarray
            Transformed feature matrix, shape (N, pca_dim).
        """
        logger.info(f"Extracting features from {len(images)} images...")
        raw_features = np.array([self.extract_single(img) for img in tqdm(images, desc="Feature extraction")])
        logger.info(f"  Raw feature shape: {raw_features.shape}")

        # Fit StandardScaler
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(raw_features)

        # Fit PCA
        actual_dim = min(self.pca_dim, scaled.shape[1], scaled.shape[0])
        self.pca = PCA(n_components=actual_dim, random_state=42)
        transformed = self.pca.fit_transform(scaled)

        self._is_fitted = True

        # Report PCA explained variance
        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        logger.info(f"  PCA: {actual_dim} components, "
                    f"explained variance = {cumulative[-1] * 100:.1f}%")
        print(f"\n  PCA Explained Variance Summary:")
        for k in [10, 50, 100, 200, 512]:
            if k <= len(cumulative):
                print(f"    Top {k:3d} components: {cumulative[k - 1] * 100:.1f}%")
        print(f"    All {actual_dim:3d} components: {cumulative[-1] * 100:.1f}%\n")

        return transformed

    def transform(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features and apply fitted scaler + PCA.

        Parameters
        ----------
        images : list[np.ndarray]
            List of grayscale images (each H, W).

        Returns
        -------
        np.ndarray
            Transformed feature matrix, shape (N, pca_dim).

        Raises
        ------
        RuntimeError
            If fit_transform() has not been called first.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit_transform() on training data first.")

        raw_features = np.array([self.extract_single(img) for img in tqdm(images, desc="Feature extraction")])
        scaled = self.scaler.transform(raw_features)
        return self.pca.transform(scaled)

    def save(self, path: str | Path) -> None:
        """Save fitted scaler and PCA to disk.

        Parameters
        ----------
        path : str | Path
            File path (will save as .joblib).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted extractor.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "scaler": self.scaler,
            "pca": self.pca,
            "pca_dim": self.pca_dim,
            "hog_orientations": self.hog_orientations,
            "hog_pixels_per_cell": self.hog_pixels_per_cell,
            "hog_cells_per_block": self.hog_cells_per_block,
            "lbp_radius": self.lbp_radius,
            "lbp_n_points": self.lbp_n_points,
        }
        joblib.dump(state, path)
        logger.info(f"  Feature extractor saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load fitted scaler and PCA from disk.

        Parameters
        ----------
        path : str | Path
            File path (.joblib).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Feature extractor not found: {path}")

        state = joblib.load(path)
        self.scaler = state["scaler"]
        self.pca = state["pca"]
        self.pca_dim = state["pca_dim"]
        self.hog_orientations = state["hog_orientations"]
        self.hog_pixels_per_cell = state["hog_pixels_per_cell"]
        self.hog_cells_per_block = state["hog_cells_per_block"]
        self.lbp_radius = state["lbp_radius"]
        self.lbp_n_points = state["lbp_n_points"]
        self._is_fitted = True
        logger.info(f"  Feature extractor loaded from {path}")


# =============================================================================
# Batch Feature Extraction from DataFrame
# =============================================================================
def extract_features_from_dataframe(
    df: "pd.DataFrame",
    extractor: ClassicalFeatureExtractor,
    preprocessor: "ChestXrayPreprocessor",
    batch_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from a DataFrame of image paths in batches.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``image_path`` and ``label_id`` columns.
    extractor : ClassicalFeatureExtractor
        Feature extractor (must be fitted for val/test).
    preprocessor : ChestXrayPreprocessor
        Image preprocessor for CLAHE + resize.
    batch_size : int
        Number of images to process per batch. Default 100.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) — feature matrix (N, D) and label array (N,).
    """
    import pandas as pd

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    failed = 0

    total = len(df)
    for start in tqdm(range(0, total, batch_size), desc="Batch feature extraction"):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]

        batch_images = []
        batch_labels = []

        for _, row in batch_df.iterrows():
            try:
                img_gray = preprocessor.preprocess_numpy(row["image_path"])
                batch_images.append(img_gray)
                batch_labels.append(int(row["label_id"]))
            except Exception as e:
                logger.warning(f"Failed to load {row['image_path']}: {e}")
                failed += 1

        if batch_images:
            features = np.array([extractor.extract_single(img) for img in batch_images])
            all_features.append(features)
            all_labels.extend(batch_labels)

    if failed > 0:
        logger.warning(f"  {failed} images failed to load.")

    X = np.vstack(all_features) if all_features else np.array([])
    y = np.array(all_labels, dtype=np.int64)

    logger.info(f"  Extracted features: X={X.shape}, y={y.shape}")
    return X, y


# =============================================================================
# __main__ Demo
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Classical Feature Extraction — Demo")
    print("=" * 60)

    extractor = ClassicalFeatureExtractor(pca_dim=512)

    # Create 10 dummy grayscale images (224×224)
    dummy_images = [np.random.randint(0, 256, (224, 224), dtype=np.uint8) for _ in range(10)]

    # Test single extraction
    single_feat = extractor.extract_single(dummy_images[0])
    print(f"\n  Single feature vector shape: {single_feat.shape}")  # Expected: (6110,)

    # Test HOG and LBP separately
    hog_feat = extractor.extract_hog(dummy_images[0])
    lbp_feat = extractor.extract_lbp(dummy_images[0])
    print(f"  HOG shape: {hog_feat.shape}")   # Expected: (6084,)
    print(f"  LBP shape: {lbp_feat.shape}")   # Expected: (26,)

    # Test fit_transform (PCA)
    X_train = extractor.fit_transform(dummy_images)
    print(f"  After PCA: {X_train.shape}")     # Expected: (10, min(512, 10))

    # Test transform on new data
    dummy_test = [np.random.randint(0, 256, (224, 224), dtype=np.uint8) for _ in range(5)]
    X_test = extractor.transform(dummy_test)
    print(f"  Test transform: {X_test.shape}")

    # Test save/load
    extractor.save("outputs/features/test_extractor.joblib")
    extractor2 = ClassicalFeatureExtractor()
    extractor2.load("outputs/features/test_extractor.joblib")
    X_test2 = extractor2.transform(dummy_test)
    print(f"  Load+transform: {X_test2.shape}")
    assert np.allclose(X_test, X_test2), "Save/load mismatch!"
    print("\n  ✓ Classical feature extraction demo complete.")

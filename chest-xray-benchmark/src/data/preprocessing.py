"""
Preprocessing Module — Chest X-ray Benchmark
==============================================
CLAHE-enhanced preprocessing and PyTorch Dataset for chest X-ray images.

Classes:
    ChestXrayPreprocessor — grayscale/RGB preprocessing with CLAHE
    ChestXrayDataset — PyTorch Dataset with split-aware augmentation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ChestXrayPreprocessor
# =============================================================================
class ChestXrayPreprocessor:
    """Preprocessing pipeline for chest X-ray images.

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    for local contrast enhancement, which is critical for CXR visibility.

    Parameters
    ----------
    img_size : int
        Target image size (square). Default 224.
    clahe_clip : float
        CLAHE clip limit. Default 2.0.
    clahe_tile : tuple[int, int]
        CLAHE tile grid size. Default (8, 8).
    """

    def __init__(
        self,
        img_size: int = 224,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
    ) -> None:
        self.img_size = img_size
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=clahe_tile
        )

    def apply_clahe(self, img_gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a grayscale image.

        Parameters
        ----------
        img_gray : np.ndarray
            Grayscale image (H, W), dtype uint8.

        Returns
        -------
        np.ndarray
            CLAHE-enhanced grayscale image (H, W), dtype uint8.
        """
        if img_gray.dtype != np.uint8:
            img_gray = (img_gray * 255).astype(np.uint8)
        return self._clahe.apply(img_gray)

    def preprocess_numpy(self, img_path: str | Path) -> np.ndarray:
        """Load image, convert to grayscale, apply CLAHE, resize.

        Parameters
        ----------
        img_path : str | Path
            Path to the image file.

        Returns
        -------
        np.ndarray
            Preprocessed grayscale image (img_size, img_size), dtype uint8.
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img = self.apply_clahe(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def preprocess_rgb(self, img_path: str | Path) -> np.ndarray:
        """Load image, apply CLAHE per-channel on grayscale, convert to RGB,
        then normalise with ImageNet statistics.

        Parameters
        ----------
        img_path : str | Path
            Path to the image file.

        Returns
        -------
        np.ndarray
            Preprocessed RGB image (img_size, img_size, 3), dtype float32,
            normalised to ImageNet statistics.
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Apply CLAHE on grayscale
        img = self.apply_clahe(img)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Convert grayscale → 3-channel RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Normalise to [0, 1] then apply ImageNet statistics
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std

        return img_float


# =============================================================================
# ChestXrayDataset (PyTorch)
# =============================================================================
class ChestXrayDataset(Dataset):
    """PyTorch Dataset for chest X-ray images with split-aware augmentation.

    Training split applies full augmentation policy:
    - Horizontal flip (p=0.5)
    - Random rotation (±10°)
    - Random affine (translate 5%)
    - ColorJitter (brightness, contrast)
    - RandomErasing (p=0.2)

    Val/test splits apply resize + normalize only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``image_path`` and ``label_id`` columns.
    split : str
        ``'train'``, ``'val'``, or ``'test'``.
    img_size : int
        Target image resolution. Default 224.
    clahe_clip : float
        CLAHE clip limit. Default 2.0.
    clahe_tile : tuple[int, int]
        CLAHE grid size. Default (8, 8).
    """

    # ImageNet normalisation statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        df: pd.DataFrame,
        split: str = "train",
        img_size: int = 224,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.split = split.lower()
        self.img_size = img_size
        self.preprocessor = ChestXrayPreprocessor(img_size, clahe_clip, clahe_tile)
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build torchvision transform pipeline based on split."""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD,
                ),
                transforms.RandomErasing(
                    p=0.2,
                    scale=(0.02, 0.1),
                ),
            ])
        else:
            # Val / Test: no augmentation
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD,
                ),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load, preprocess, and augment a single image.

        Parameters
        ----------
        idx : int
            Index into the DataFrame.

        Returns
        -------
        tuple[torch.Tensor, int]
            (image_tensor [3, img_size, img_size], label_id)
        """
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label_id = int(row["label_id"])

        try:
            # Load and apply CLAHE
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise FileNotFoundError(f"Cannot read: {img_path}")

            img_gray = self.preprocessor.apply_clahe(img_gray)
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            # Convert to PIL for torchvision transforms
            pil_img = Image.fromarray(img_rgb)
            tensor = self.transform(pil_img)

        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}. Returning zeros.")
            tensor = torch.zeros(3, self.img_size, self.img_size)

        return tensor, label_id

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array (for computing class weights).

        Returns
        -------
        np.ndarray
            Array of label IDs, shape (N,).
        """
        return self.df["label_id"].values.astype(np.int64)


if __name__ == "__main__":
    print("=" * 60)
    print("  Preprocessing Module — Demo")
    print("=" * 60)

    # Demo with dummy data
    preprocessor = ChestXrayPreprocessor(img_size=224, clahe_clip=2.0)
    print(f"\n  Preprocessor: img_size={preprocessor.img_size}, "
          f"CLAHE clip={preprocessor.clahe_clip}, tile={preprocessor.clahe_tile}")

    # Create a dummy grayscale image
    dummy_gray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    enhanced = preprocessor.apply_clahe(dummy_gray)
    print(f"  CLAHE applied: input shape={dummy_gray.shape}, output shape={enhanced.shape}")

    # Demo Dataset
    dummy_df = pd.DataFrame({
        "image_path": [f"/dummy/img_{i}.png" for i in range(10)],
        "label_id": [i % 6 for i in range(10)],
    })
    dataset = ChestXrayDataset(dummy_df, split="train", img_size=224)
    print(f"  Dataset: {len(dataset)} samples, split='train'")
    print(f"  Labels: {dataset.get_labels()}")
    print("\n  ✓ Preprocessing module demo complete.")

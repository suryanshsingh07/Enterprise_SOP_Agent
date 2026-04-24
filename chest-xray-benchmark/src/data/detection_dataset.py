"""
Detection Dataset — Chest X-ray Benchmark
============================================
PyTorch Dataset for object detection with bounding box annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import pandas as pd


class DetectionDataset(Dataset):
    """PyTorch Dataset for chest X-ray detection tasks.

    Loads images and their bounding box annotations for Faster/Mask R-CNN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: image_path, label_id.
        Optional bbox columns: x_min, y_min, x_max, y_max.
    img_size : int
        Target image size. Default 224.
    transforms : callable, optional
        Albumentations-style transforms.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 224,
        transforms: Optional[Any] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.transforms = transforms
        self.has_bbox = all(c in df.columns for c in ["x_min", "y_min", "x_max", "y_max"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Load image and target.

        Parameters
        ----------
        idx : int
            Index.

        Returns
        -------
        tuple[torch.Tensor, dict]
            (image, target) where target has 'boxes', 'labels', 'area', 'iscrowd'.
        """
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label_id = int(row["label_id"])

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Bounding box
        if self.has_bbox:
            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            x_max = float(row["x_max"])
            y_max = float(row["y_max"])
        else:
            margin = int(self.img_size * 0.05)
            x_min = float(margin)
            y_min = float(margin)
            x_max = float(self.img_size - margin)
            y_max = float(self.img_size - margin)

        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.tensor([label_id], dtype=torch.int64)
        area = (x_max - x_min) * (y_max - y_min)
        area = torch.tensor([area], dtype=torch.float32)
        iscrowd = torch.zeros(1, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx]),
        }

        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, target


def collate_fn(batch: List) -> Tuple:
    """Custom collate function for detection DataLoader.

    Parameters
    ----------
    batch : list
        List of (image, target) tuples.

    Returns
    -------
    tuple
        (list[Tensor], list[dict])
    """
    return tuple(zip(*batch))

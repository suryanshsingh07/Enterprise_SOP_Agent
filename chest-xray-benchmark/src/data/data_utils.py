"""
Data Utilities — Chest X-ray Benchmark
========================================
Helper functions for image I/O, path validation, and data integrity checks.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Valid image extensions
VALID_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def validate_image(path: str | Path) -> bool:
    """Check if a file is a valid, openable image.

    Parameters
    ----------
    path : str | Path
        Path to the image file.

    Returns
    -------
    bool
        True if the file exists and can be opened as an image.
    """
    path = Path(path)
    if not path.exists():
        return False
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def filter_valid_images(df: pd.DataFrame, image_col: str = "image_path") -> pd.DataFrame:
    """Remove rows with missing or corrupt images.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an image path column.
    image_col : str
        Name of the column containing image paths.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only valid images.
    """
    valid_mask = df[image_col].apply(lambda p: Path(p).exists())
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"Removed {n_invalid:,} rows with missing images.")
    return df[valid_mask].reset_index(drop=True)


def compute_file_hash(path: str | Path, algorithm: str = "md5") -> str:
    """Compute hash of a file for deduplication.

    Parameters
    ----------
    path : str | Path
        File path.
    algorithm : str
        Hash algorithm ('md5', 'sha256').

    Returns
    -------
    str
        Hex digest of the file hash.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_duplicates(df: pd.DataFrame, image_col: str = "image_path") -> pd.DataFrame:
    """Remove duplicate images based on file hash.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an image path column.
    image_col : str
        Name of the column containing image paths.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    logger.info("Computing file hashes for deduplication...")
    hashes: List[Optional[str]] = []
    for path in df[image_col]:
        if Path(path).exists():
            hashes.append(compute_file_hash(path))
        else:
            hashes.append(None)

    df = df.copy()
    df["_hash"] = hashes
    n_before = len(df)
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"]).reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed:,} duplicate images.")
    return df


def get_image_stats(path: str | Path) -> dict:
    """Get basic statistics about an image file.

    Parameters
    ----------
    path : str | Path
        Image file path.

    Returns
    -------
    dict
        Dictionary with keys: width, height, channels, mode, size_kb.
    """
    path = Path(path)
    with Image.open(path) as img:
        w, h = img.size
        mode = img.mode
        channels = len(img.getbands())

    size_kb = path.stat().st_size / 1024

    return {
        "width": w,
        "height": h,
        "channels": channels,
        "mode": mode,
        "size_kb": round(size_kb, 1),
    }


def dataset_summary_stats(df: pd.DataFrame, n_sample: int = 100) -> dict:
    """Compute dataset-level image statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``image_path`` column.
    n_sample : int
        Number of images to sample for computing statistics.

    Returns
    -------
    dict
        Dataset-level statistics (mean size, aspect ratio, etc.).
    """
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    widths, heights, sizes = [], [], []

    for path in sample["image_path"]:
        if not Path(path).exists():
            continue
        stats = get_image_stats(path)
        widths.append(stats["width"])
        heights.append(stats["height"])
        sizes.append(stats["size_kb"])

    return {
        "n_images": len(df),
        "sampled": len(widths),
        "mean_width": round(np.mean(widths), 1) if widths else 0,
        "mean_height": round(np.mean(heights), 1) if heights else 0,
        "mean_size_kb": round(np.mean(sizes), 1) if sizes else 0,
        "min_size_kb": round(min(sizes), 1) if sizes else 0,
        "max_size_kb": round(max(sizes), 1) if sizes else 0,
    }

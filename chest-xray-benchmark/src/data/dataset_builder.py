"""
Dataset Builder — Chest X-ray Benchmark
=========================================
Unified dataset construction from 6 heterogeneous sources into a single
6-class schema with stratified splitting, class balancing, and ImageNet
folder structure export.

Classes:
    DatasetBuilder — main builder accepting source configs
Functions:
    stratified_split — 70/15/15 preserving class distribution
    balance_classes — oversample / undersample / cap
    copy_to_imagenet_structure — export to class-folder layout
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tabulate import tabulate
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# Unified 6-Class Label Schema
# =============================================================================
UNIFIED_CLASSES: Dict[int, str] = {
    0: "Normal",
    1: "COVID-19",
    2: "Pneumonia",
    3: "Tuberculosis",
    4: "Lung_Cancer",
    5: "Fibrosis",
}

CLASS_NAMES: List[str] = list(UNIFIED_CLASSES.values())
NUM_CLASSES: int = len(UNIFIED_CLASSES)

# Reverse lookup: class name → id
CLASS_TO_ID: Dict[str, int] = {v: k for k, v in UNIFIED_CLASSES.items()}

# =============================================================================
# Label Mapping — source labels → unified label IDs
# =============================================================================
LABEL_MAP: Dict[str, int] = {
    # ── COVID-QU-Ex ───────────────────────────────────────────────────────────
    "COVID-19": 1,
    "covid-19": 1,
    "covid": 1,
    "Non-COVID": 2,          # Non-COVID respiratory = Pneumonia
    "non-covid": 2,
    "Normal": 0,
    "normal": 0,
    "NORMAL": 0,

    # ── CheXpert ──────────────────────────────────────────────────────────────
    "No Finding": 0,
    "no finding": 0,
    "Consolidation": 2,      # Consolidation → Pneumonia
    "consolidation": 2,
    "Lung Opacity": 5,       # Opacity → Fibrosis
    "lung opacity": 5,
    "Pleural Effusion": 5,   # Effusion → Fibrosis
    "pleural effusion": 5,
    "Cardiomegaly": 0,       # Excluded — map to Normal (skip these)
    "Atelectasis": 2,        # Atelectasis → Pneumonia

    # ── NIH ChestX-ray14 ─────────────────────────────────────────────────────
    "Pneumonia": 2,
    "pneumonia": 2,
    "Fibrosis": 5,
    "fibrosis": 5,
    "Infiltration": 2,       # Infiltration → Pneumonia
    "infiltration": 2,
    "Mass": 4,               # Mass → Lung Cancer
    "mass": 4,
    "Nodule": 4,             # Nodule → Lung Cancer
    "nodule": 4,
    "Effusion": 5,           # Effusion → Fibrosis

    # ── VinDr-CXR ─────────────────────────────────────────────────────────────
    "No finding": 0,
    "Tuberculosis": 3,
    "tuberculosis": 3,
    "Lung tumor": 4,
    "lung tumor": 4,
    "Pulmonary fibrosis": 5,
    "pulmonary fibrosis": 5,

    # ── Shenzhen TB ───────────────────────────────────────────────────────────
    "tb": 3,
    "TB": 3,

    # ── LIDC-IDRI ─────────────────────────────────────────────────────────────
    "malignant": 4,          # Malignancy ≥ 3 → Lung Cancer
    "Malignant": 4,
    "benign": 0,             # Benign nodule → Normal
    "Benign": 0,

    # ── Generic fallbacks ─────────────────────────────────────────────────────
    "Lung_Cancer": 4,
    "lung_cancer": 4,
    "COVID": 1,
}

# Image extensions considered valid
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# =============================================================================
# DatasetBuilder
# =============================================================================
class DatasetBuilder:
    """Builds a unified DataFrame from multiple heterogeneous CXR sources.

    Each source can be either:
    - ``imagenet_folder``: subdirectory per class (``root/ClassName/image.png``)
    - ``csv_manifest``: CSV with image path and label columns

    Parameters
    ----------
    source_configs : list[dict]
        List of dicts, each with keys:
        - ``name`` (str): dataset identifier
        - ``root`` (str): root directory
        - ``format`` (str): ``'imagenet_folder'`` or ``'csv_manifest'``
        - ``manifest`` (str, optional): path to CSV for csv_manifest format
        - ``image_col`` (str, optional): column name for image paths
        - ``label_col`` (str, optional): column name for labels
        - ``image_dir`` (str, optional): directory containing images
        - ``classes`` (list[dict], optional): for folder format
    """

    def __init__(self, source_configs: List[Dict[str, Any]]) -> None:
        self.source_configs = source_configs
        self.records: List[Dict[str, Any]] = []

    def build(self) -> pd.DataFrame:
        """Build the unified dataset DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: [image_path, label_id, label_name, source]
        """
        self.records = []

        for cfg in self.source_configs:
            name = cfg["name"]
            fmt = cfg["format"]
            logger.info(f"Processing source: {name} (format={fmt})")

            try:
                if fmt == "imagenet_folder":
                    records = self._load_imagenet_folder(cfg)
                elif fmt == "csv_manifest":
                    records = self._load_csv_manifest(cfg)
                else:
                    logger.warning(f"Unknown format '{fmt}' for {name}, skipping.")
                    continue

                self.records.extend(records)
                logger.info(f"  → Loaded {len(records):,} images from {name}")

            except FileNotFoundError as e:
                logger.warning(f"  ⚠ Source '{name}' not found: {e}. Skipping.")
            except Exception as e:
                logger.error(f"  ✗ Error loading '{name}': {e}")

        if not self.records:
            logger.warning("No records loaded from any source!")
            return pd.DataFrame(columns=["image_path", "label_id", "label_name", "source"])

        df = pd.DataFrame(self.records)
        df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

        self._print_summary(df)
        return df

    # ── Private Loaders ───────────────────────────────────────────────────────

    def _load_imagenet_folder(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load from subdirectory-per-class structure."""
        root = Path(cfg["root"])
        name = cfg["name"]
        records: List[Dict[str, Any]] = []

        if "classes" in cfg:
            # Explicit class mapping provided
            for cls_info in cfg["classes"]:
                src_name = cls_info["source_name"]
                target_id = cls_info["target_id"]
                cls_dir = root / src_name

                if not cls_dir.is_dir():
                    logger.warning(f"  Class dir not found: {cls_dir}")
                    continue

                for img_file in cls_dir.iterdir():
                    if img_file.suffix.lower() in VALID_EXTENSIONS:
                        records.append({
                            "image_path": str(img_file.resolve()),
                            "label_id": target_id,
                            "label_name": UNIFIED_CLASSES[target_id],
                            "source": name,
                        })
        else:
            # Auto-detect subdirectories, map via LABEL_MAP
            for cls_dir in sorted(root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls_name = cls_dir.name
                label_id = LABEL_MAP.get(cls_name)
                if label_id is None:
                    logger.warning(f"  Unknown class '{cls_name}' in {name}, skipping.")
                    continue

                for img_file in cls_dir.iterdir():
                    if img_file.suffix.lower() in VALID_EXTENSIONS:
                        records.append({
                            "image_path": str(img_file.resolve()),
                            "label_id": label_id,
                            "label_name": UNIFIED_CLASSES[label_id],
                            "source": name,
                        })

        return records

    def _load_csv_manifest(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load from CSV manifest with image path and label columns."""
        manifest_path = Path(cfg["manifest"])
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        image_col = cfg.get("image_col", "image_path")
        label_col = cfg.get("label_col", "label")
        image_dir = Path(cfg.get("image_dir", cfg["root"]))
        name = cfg["name"]
        records: List[Dict[str, Any]] = []

        csv_df = pd.read_csv(manifest_path)

        if image_col not in csv_df.columns:
            raise ValueError(f"Column '{image_col}' not in {manifest_path}")
        if label_col not in csv_df.columns:
            raise ValueError(f"Column '{label_col}' not in {manifest_path}")

        for _, row in csv_df.iterrows():
            raw_label = str(row[label_col]).strip()

            # Handle pipe-separated multi-labels (NIH): take first mapped label
            if "|" in raw_label:
                sub_labels = [s.strip() for s in raw_label.split("|")]
                mapped = [LABEL_MAP[s] for s in sub_labels if s in LABEL_MAP]
                if not mapped:
                    continue
                label_id = mapped[0]  # Take dominant (first matched) finding
            else:
                label_id = LABEL_MAP.get(raw_label)

            if label_id is None:
                continue

            img_path = image_dir / str(row[image_col])
            records.append({
                "image_path": str(img_path.resolve()),
                "label_id": label_id,
                "label_name": UNIFIED_CLASSES[label_id],
                "source": name,
            })

        return records

    # ── Summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        """Print a formatted summary table of the dataset."""
        print("\n" + "=" * 60)
        print("  UNIFIED DATASET SUMMARY")
        print("=" * 60)

        # Per-class counts
        class_counts = df.groupby(["label_id", "label_name"]).size().reset_index(name="count")
        class_counts = class_counts.sort_values("label_id")

        table = []
        for _, row in class_counts.iterrows():
            pct = 100.0 * row["count"] / len(df)
            table.append([row["label_id"], row["label_name"], f"{row['count']:,}", f"{pct:.1f}%"])

        table.append(["", "TOTAL", f"{len(df):,}", "100.0%"])

        print(tabulate(table, headers=["ID", "Class", "Count", "%"], tablefmt="rounded_grid"))

        # Per-source counts
        source_counts = df["source"].value_counts()
        src_table = [[src, f"{cnt:,}"] for src, cnt in source_counts.items()]
        print("\n  Per-Source Counts:")
        print(tabulate(src_table, headers=["Source", "Count"], tablefmt="rounded_grid"))
        print()


# =============================================================================
# Stratified Split
# =============================================================================
def stratified_split(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test preserving class distribution.

    Uses two rounds of ``StratifiedShuffleSplit``:
    1. Split off test set (``test`` fraction)
    2. Split remaining into train and val

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``label_id`` column.
    train : float
        Training fraction (default 0.70).
    val : float
        Validation fraction (default 0.15).
    test : float
        Test fraction (default 0.15).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    assert abs(train + val + test - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train + val + test:.4f}"

    labels = df["label_id"].values

    # Round 1: split off test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(df, labels))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Round 2: split train_val into train and val
    val_relative = val / (train + val)
    labels_tv = train_val_df["label_id"].values
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val_df, labels_tv))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Verification table
    _print_split_verification(train_df, val_df, test_df)

    return train_df, val_df, test_df


def _print_split_verification(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print a verification table showing class distribution across splits."""
    print("\n" + "=" * 70)
    print("  STRATIFIED SPLIT VERIFICATION")
    print("=" * 70)

    table = []
    for label_id in sorted(UNIFIED_CLASSES.keys()):
        name = UNIFIED_CLASSES[label_id]
        n_train = (train_df["label_id"] == label_id).sum()
        n_val = (val_df["label_id"] == label_id).sum()
        n_test = (test_df["label_id"] == label_id).sum()
        total = n_train + n_val + n_test

        if total > 0:
            pct_train = 100.0 * n_train / total
            pct_val = 100.0 * n_val / total
            pct_test = 100.0 * n_test / total
        else:
            pct_train = pct_val = pct_test = 0.0

        table.append([
            label_id, name,
            f"{n_train:,}", f"{pct_train:.1f}%",
            f"{n_val:,}", f"{pct_val:.1f}%",
            f"{n_test:,}", f"{pct_test:.1f}%",
            f"{total:,}",
        ])

    table.append([
        "", "TOTAL",
        f"{len(train_df):,}", f"{100 * len(train_df) / (len(train_df) + len(val_df) + len(test_df)):.1f}%",
        f"{len(val_df):,}", f"{100 * len(val_df) / (len(train_df) + len(val_df) + len(test_df)):.1f}%",
        f"{len(test_df):,}", f"{100 * len(test_df) / (len(train_df) + len(val_df) + len(test_df)):.1f}%",
        f"{len(train_df) + len(val_df) + len(test_df):,}",
    ])

    headers = ["ID", "Class", "Train", "%", "Val", "%", "Test", "%", "Total"]
    print(tabulate(table, headers=headers, tablefmt="rounded_grid"))
    print()


# =============================================================================
# Class Balancing
# =============================================================================
def balance_classes(
    df: pd.DataFrame,
    strategy: Literal["oversample", "undersample", "cap"] = "oversample",
    max_per_class: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Balance class distribution using specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with ``label_id`` column.
    strategy : str
        ``'oversample'`` — sample with replacement to match largest class.
        ``'undersample'`` — sample down to match smallest class.
        ``'cap'`` — cap at ``max_per_class`` per class.
    max_per_class : int
        Maximum samples per class when strategy is ``'cap'``.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Balanced DataFrame.
    """
    rng = np.random.RandomState(seed)
    class_counts = df["label_id"].value_counts()

    print("\n  Class Distribution BEFORE Balancing:")
    for cid in sorted(class_counts.index):
        print(f"    {UNIFIED_CLASSES[cid]:15s}: {class_counts[cid]:>7,}")

    if strategy == "oversample":
        target_count = class_counts.max()
    elif strategy == "undersample":
        target_count = class_counts.min()
    elif strategy == "cap":
        target_count = max_per_class
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'oversample', 'undersample', or 'cap'.")

    balanced_parts: List[pd.DataFrame] = []

    for label_id in sorted(class_counts.index):
        class_df = df[df["label_id"] == label_id]
        n = len(class_df)

        if n == target_count:
            balanced_parts.append(class_df)
        elif n < target_count:
            # Need more samples — oversample with replacement
            extra_idx = rng.choice(class_df.index, size=target_count - n, replace=True)
            balanced_parts.append(class_df)
            balanced_parts.append(df.loc[extra_idx])
        else:
            # Need fewer samples — subsample without replacement
            sampled_idx = rng.choice(class_df.index, size=target_count, replace=False)
            balanced_parts.append(df.loc[sampled_idx])

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"\n  Class Distribution AFTER Balancing (strategy={strategy}):")
    after_counts = balanced_df["label_id"].value_counts()
    for cid in sorted(after_counts.index):
        print(f"    {UNIFIED_CLASSES[cid]:15s}: {after_counts[cid]:>7,}")
    print(f"    {'TOTAL':15s}: {len(balanced_df):>7,}\n")

    return balanced_df


# =============================================================================
# Copy to ImageNet Folder Structure
# =============================================================================
def copy_to_imagenet_structure(
    df: pd.DataFrame,
    split_name: str,
    output_dir: str | Path,
) -> None:
    """Copy images into ImageNet-style directory layout.

    Creates:  ``output_dir/split_name/ClassName/filename.ext``

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``image_path`` and ``label_name`` columns.
    split_name : str
        E.g. ``'train'``, ``'val'``, ``'test'``.
    output_dir : str | Path
        Root output directory.
    """
    output_dir = Path(output_dir)
    skipped = 0
    copied = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        src = Path(row["image_path"])
        class_name = row["label_name"]
        dst_dir = output_dir / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if dst.exists():
            skipped += 1
            continue

        if not src.exists():
            missing += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    logger.info(
        f"  {split_name}: copied={copied:,}, skipped={skipped:,}, missing={missing:,}"
    )


# =============================================================================
# Save Manifests
# =============================================================================
def save_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save train/val/test DataFrames as CSV manifests and class stats as JSON.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames.
    output_dir : str | Path
        Directory to write manifests to.
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Class stats
    stats = {
        "classes": UNIFIED_CLASSES,
        "splits": {
            "train": train_df["label_id"].value_counts().to_dict(),
            "val": val_df["label_id"].value_counts().to_dict(),
            "test": test_df["label_id"].value_counts().to_dict(),
        },
        "totals": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }

    with open(output_dir / "class_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Manifests saved to {output_dir}")


# =============================================================================
# __main__ Demo
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Builder — Demo with Dummy Data")
    print("=" * 60)

    # Create dummy records to demonstrate the pipeline
    rng = np.random.RandomState(42)
    dummy_records = []
    for label_id, label_name in UNIFIED_CLASSES.items():
        n_samples = rng.randint(50, 200)
        for i in range(n_samples):
            dummy_records.append({
                "image_path": f"/dummy/data/{label_name}/img_{i:04d}.png",
                "label_id": label_id,
                "label_name": label_name,
                "source": "dummy",
            })

    df = pd.DataFrame(dummy_records)
    print(f"\nTotal dummy images: {len(df):,}")

    # Stratified split
    train_df, val_df, test_df = stratified_split(df, train=0.70, val=0.15, test=0.15)

    # Balance training set
    train_balanced = balance_classes(train_df, strategy="oversample")

    # Show final summary
    print("\n  Final Summary:")
    print(f"    Train (balanced): {len(train_balanced):,}")
    print(f"    Val:              {len(val_df):,}")
    print(f"    Test:             {len(test_df):,}")
    print("\n  ✓ Dataset builder demo complete.")

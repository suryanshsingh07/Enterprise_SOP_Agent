#!/usr/bin/env python3
"""
01 — Build Dataset
===================
Script to build the unified 6-class chest X-ray dataset from raw sources.

Usage:
    python scripts/01_build_dataset.py
    python scripts/01_build_dataset.py --config configs/dataset_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import (
    DatasetBuilder,
    balance_classes,
    copy_to_imagenet_structure,
    save_manifests,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build unified chest X-ray dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--balance",
        type=str,
        default="oversample",
        choices=["oversample", "undersample", "cap"],
        help="Class balancing strategy",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        help="Max images per class (for 'cap' strategy)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to ImageNet folder structure",
    )
    return parser.parse_args()


def build_source_configs(config: dict) -> list[dict]:
    """Convert YAML config into list of source config dicts for DatasetBuilder.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.

    Returns
    -------
    list[dict]
        List of source configs ready for DatasetBuilder.
    """
    sources = []
    for name, dataset_cfg in config.get("datasets", {}).items():
        src = {"name": name}
        src["root"] = dataset_cfg.get("root", "")
        src["format"] = dataset_cfg.get("format", "imagenet_folder")

        if src["format"] == "csv_manifest":
            src["manifest"] = dataset_cfg.get("manifest", "")
            src["image_col"] = dataset_cfg.get("image_col", "image_path")
            src["label_col"] = dataset_cfg.get("label_col", "label")
            src["image_dir"] = dataset_cfg.get("image_dir", src["root"])

        if "classes" in dataset_cfg:
            src["classes"] = dataset_cfg["classes"]

        sources.append(src)

    return sources


def main() -> None:
    """Main dataset building pipeline."""
    args = parse_args()

    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — DATASET BUILDER")
    print("=" * 60)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"\n  Config loaded from: {config_path}")
    else:
        print(f"\n  ⚠ Config not found: {config_path}")
        print("  Using default empty config. Place datasets in data/raw/")
        config = {"datasets": {}}

    # Build source configs
    source_configs = build_source_configs(config)

    if not source_configs:
        print("\n  No dataset sources configured. Exiting.")
        print("  → Edit configs/dataset_config.yaml and re-run.")
        return

    # Build unified dataset
    builder = DatasetBuilder(source_configs)
    df = builder.build()

    if df.empty:
        print("\n  ⚠ No images found. Check your data paths.")
        return

    # Stratified split
    split_cfg = config.get("split", {})
    train_df, val_df, test_df = stratified_split(
        df,
        train=split_cfg.get("train", 0.70),
        val=split_cfg.get("val", 0.15),
        test=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", 42),
    )

    # Balance training set
    train_df = balance_classes(
        train_df,
        strategy=args.balance,
        max_per_class=args.max_per_class,
    )

    # Save manifests
    manifests_dir = config.get("output", {}).get("manifests_dir", "data/manifests")
    save_manifests(train_df, val_df, test_df, manifests_dir)

    # Optionally copy to ImageNet structure
    if args.copy_images:
        output_dir = Path(args.output_dir)
        print("\n  Copying images to ImageNet structure...")
        copy_to_imagenet_structure(train_df, "train", output_dir)
        copy_to_imagenet_structure(val_df, "val", output_dir)
        copy_to_imagenet_structure(test_df, "test", output_dir)

    print("\n" + "=" * 60)
    print("  ✓ Dataset build complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

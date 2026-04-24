"""
Annotation Converter — Chest X-ray Benchmark
===============================================
Convert between annotation formats: COCO JSON, YOLO TXT, and CSV.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_coco_annotations(
    df: pd.DataFrame,
    image_dir: str | Path,
    output_path: str | Path,
    img_size: int = 224,
) -> None:
    """Create COCO-format annotations from a DataFrame.

    For classification-only data, creates a full-image bounding box.
    For data with bounding boxes, uses the provided boxes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: image_path, label_id, label_name.
        Optional bbox columns: x_min, y_min, x_max, y_max.
    image_dir : str | Path
        Base image directory.
    output_path : str | Path
        Output JSON path.
    img_size : int
        Image size for default bounding boxes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.data.dataset_builder import UNIFIED_CLASSES

    coco: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Categories
    for cid, cname in UNIFIED_CLASSES.items():
        coco["categories"].append({
            "id": cid,
            "name": cname,
            "supercategory": "disease",
        })

    ann_id = 0
    for img_id, (_, row) in enumerate(df.iterrows()):
        filename = Path(row["image_path"]).name

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": img_size,
            "height": img_size,
        })

        # Use provided bbox or full-image bbox
        if all(c in df.columns for c in ["x_min", "y_min", "x_max", "y_max"]):
            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            w = float(row["x_max"]) - x_min
            h = float(row["y_max"]) - y_min
        else:
            # Full-image bbox with small margin
            margin = int(img_size * 0.05)
            x_min = float(margin)
            y_min = float(margin)
            w = float(img_size - 2 * margin)
            h = float(img_size - 2 * margin)

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": int(row["label_id"]),
            "bbox": [x_min, y_min, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(f"  COCO annotations saved: {output_path} ({len(coco['images'])} images)")


def create_yolo_annotations(
    df: pd.DataFrame,
    image_dir: str | Path,
    output_dir: str | Path,
    img_size: int = 224,
) -> None:
    """Create YOLO-format annotation text files.

    Creates one .txt file per image with format:
    ``class_id cx cy w h`` (all normalised to [0, 1])

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with image_path, label_id columns.
    image_dir : str | Path
        Source image directory.
    output_dir : str | Path
        Output directory for .txt files.
    img_size : int
        Image size for normalisation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        img_name = Path(row["image_path"]).stem
        label_id = int(row["label_id"])

        if all(c in df.columns for c in ["x_min", "y_min", "x_max", "y_max"]):
            x_min = float(row["x_min"]) / img_size
            y_min = float(row["y_min"]) / img_size
            x_max = float(row["x_max"]) / img_size
            y_max = float(row["y_max"]) / img_size
        else:
            margin = 0.05
            x_min = margin
            y_min = margin
            x_max = 1.0 - margin
            y_max = 1.0 - margin

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        txt_path = output_dir / f"{img_name}.txt"
        with open(txt_path, "w") as f:
            f.write(f"{label_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    logger.info(f"  YOLO annotations saved: {output_dir} ({len(df)} files)")


def coco_to_yolo(
    coco_json_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Convert COCO JSON annotations to YOLO format.

    Parameters
    ----------
    coco_json_path : str | Path
        Path to COCO JSON file.
    output_dir : str | Path
        Directory to save YOLO .txt files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image lookup
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    img_anns: Dict[int, List] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    for img_id, anns in img_anns.items():
        img_info = img_lookup[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_name = Path(img_info["file_name"]).stem

        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        txt_path = output_dir / f"{img_name}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    logger.info(f"  Converted {len(img_anns)} images from COCO to YOLO")

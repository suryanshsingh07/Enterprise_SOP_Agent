#!/usr/bin/env python3
"""
09 — Evaluate All Models
=========================
Run inference on the test set for every saved model,
compute all metrics, and generate benchmark comparison.

Usage:
    python scripts/09_evaluate_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import CLASS_NAMES, NUM_CLASSES
from src.data.preprocessing import ChestXrayDataset
from src.evaluation.evaluator import (
    BenchmarkEvaluator,
    cross_model_comparison_plot,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.training.utils import set_seed, get_device


def evaluate_dl_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on test set.

    Returns
    -------
    tuple
        (y_true, y_pred, y_proba)
    """
    model.eval()
    model.to(device)

    all_labels, all_preds, all_proba = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_proba.append(proba)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_proba),
    )


def main() -> None:
    """Evaluate all saved models on the test set."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — EVALUATION")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    manifests_dir = Path("data/manifests")
    checkpoint_dir = Path("outputs/checkpoints")
    classical_dir = Path("outputs/classical")
    eval_dir = Path("outputs/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load test manifest
    test_csv = manifests_dir / "test.csv"
    if not test_csv.exists():
        print(f"\n  ⚠ {test_csv} not found. Run 01_build_dataset.py first.")
        return

    test_df = pd.read_csv(test_csv)
    print(f"\n  Test set: {len(test_df):,} images")

    evaluator = BenchmarkEvaluator(class_names=CLASS_NAMES)

    # ── Classical Models ──────────────────────────────────────────────────────
    features_dir = Path("outputs/features")
    if (features_dir / "X_test.npy").exists():
        import joblib
        X_test = np.load(features_dir / "X_test.npy")
        y_test = np.load(features_dir / "y_test.npy")

        for model_file in classical_dir.glob("*.joblib"):
            model_name = model_file.stem
            if model_name.endswith("_metrics") or model_name == "feature_extractor":
                continue
            print(f"\n  Evaluating: {model_name}")
            model = joblib.load(model_file)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            metrics = evaluator.evaluate_model(model_name, y_test, y_pred, y_proba)
            plot_confusion_matrix(
                y_test, y_pred, CLASS_NAMES, model_name,
                eval_dir / f"cm_{model_name}.png",
            )
            if y_proba is not None:
                plot_roc_curves(
                    y_test, y_proba, CLASS_NAMES, model_name,
                    eval_dir / f"roc_{model_name}.png",
                )

    # ── DL Models ─────────────────────────────────────────────────────────────
    test_ds = ChestXrayDataset(test_df, split="test", img_size=224)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Map checkpoint names to model classes
    dl_models = {
        "efficientnet_b3_best.pth": ("EfficientNet-B3", "src.models.cnn_models", "EfficientNetChest"),
        "resnet50_best.pth": ("ResNet-50", "src.models.cnn_models", "ResNet50Chest"),
        "swin_tiny_best.pth": ("Swin-Tiny", "src.models.transformer_models", "SwinChest"),
        "vit_base16_best.pth": ("ViT-B/16", "src.models.transformer_models", "ViTChest"),
    }

    for ckpt_file, (display_name, module_path, class_name) in dl_models.items():
        ckpt_path = checkpoint_dir / ckpt_file
        if not ckpt_path.exists():
            print(f"  Skipping {display_name}: checkpoint not found")
            continue

        print(f"\n  Evaluating: {display_name}")
        import importlib
        mod = importlib.import_module(module_path)
        model_cls = getattr(mod, class_name)
        model = model_cls(n_classes=NUM_CLASSES, freeze_backbone=False)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        y_true, y_pred, y_proba = evaluate_dl_model(model, test_loader, device)
        metrics = evaluator.evaluate_model(display_name, y_true, y_pred, y_proba)

        plot_confusion_matrix(
            y_true, y_pred, CLASS_NAMES, display_name,
            eval_dir / f"cm_{display_name.replace('/', '_')}.png",
        )
        plot_roc_curves(
            y_true, y_proba, CLASS_NAMES, display_name,
            eval_dir / f"roc_{display_name.replace('/', '_')}.png",
        )
        plot_precision_recall_curves(
            y_true, y_proba, CLASS_NAMES, display_name,
            eval_dir / f"pr_{display_name.replace('/', '_')}.png",
        )

    # ── Cross-Model Comparison ────────────────────────────────────────────────
    if evaluator.results:
        cross_model_comparison_plot(
            evaluator.results, metric="macro_f1",
            save_path=eval_dir / "model_comparison_f1.png",
        )

        # Save all results
        evaluator.save_results(eval_dir / "all_metrics.json")

        # Also save to experiments
        experiments_dir = Path("experiments")
        experiments_dir.mkdir(parents=True, exist_ok=True)
        evaluator.save_results(experiments_dir / "all_metrics.json")

        # Print comparison table
        print("\n" + "=" * 70)
        print("  FINAL BENCHMARK COMPARISON TABLE")
        print("=" * 70)
        table = evaluator.comparison_table()
        print(table.to_string())

    print("\n" + "=" * 60)
    print("  ✓ Evaluation complete!")
    print(f"  Results saved to: {eval_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

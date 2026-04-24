#!/usr/bin/env python3
"""
03 — Train Classical ML Models
================================
Train SVM, Logistic Regression, XGBoost, and AdaBoost on HOG+LBP+PCA features.

Usage:
    python scripts/03_train_classical.py
    python scripts/03_train_classical.py --tune-xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import CLASS_NAMES
from src.models.classical_models import (
    build_adaboost,
    build_logistic_regression,
    build_svm,
    build_xgboost,
    train_classical_model,
    tune_xgboost_optuna,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train classical ML models")
    parser.add_argument(
        "--features-dir",
        type=str,
        default="outputs/features",
        help="Directory with saved feature arrays",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/classical",
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--tune-xgboost",
        action="store_true",
        help="Run Optuna tuning for XGBoost",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    return parser.parse_args()


def main() -> None:
    """Train all classical models and produce comparison."""
    args = parse_args()

    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — CLASSICAL ML TRAINING")
    print("=" * 60)

    features_dir = Path(args.features_dir)
    save_dir = Path(args.save_dir)

    # Load features
    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    for f in required_files:
        if not (features_dir / f).exists():
            print(f"\n  ⚠ Feature file not found: {features_dir / f}")
            print("  → Run 02_extract_features.py first.")
            return

    X_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    X_val = np.load(features_dir / "X_val.npy")
    y_val = np.load(features_dir / "y_val.npy")

    print(f"\n  Features loaded:")
    print(f"    Train: {X_train.shape[0]:,} × {X_train.shape[1]}")
    print(f"    Val:   {X_val.shape[0]:,} × {X_val.shape[1]}")

    # Optional: Optuna tuning for XGBoost
    if args.tune_xgboost:
        best_params = tune_xgboost_optuna(
            X_train, y_train, X_val, y_val,
            n_trials=args.optuna_trials,
        )
        # Save best params
        params_path = save_dir / "xgboost_best_params.json"
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"  Best params saved: {params_path}")

    # Build all models
    models = {
        "SVM": build_svm(C=10.0, kernel="rbf"),
        "LogisticRegression": build_logistic_regression(C=1.0),
        "XGBoost": build_xgboost(n_classes=6),
        "AdaBoost": build_adaboost(),
    }

    # Train and evaluate each
    all_results = {}
    for name, model in models.items():
        fitted_model, metrics = train_classical_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=name,
            save_dir=save_dir,
            class_names=CLASS_NAMES,
        )
        all_results[name] = metrics

    # Save comparison CSV
    comparison_df = pd.DataFrame(all_results).T
    comparison_df.index.name = "model"
    comparison_df = comparison_df.sort_values("macro_f1", ascending=False)

    csv_path = save_dir / "classical_comparison.csv"
    comparison_df.to_csv(csv_path)

    # Save as JSON for experiments
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / "classical_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final comparison
    print("\n" + "=" * 70)
    print("  CLASSICAL ML — FINAL COMPARISON (sorted by Macro F1)")
    print("=" * 70)
    print(comparison_df.to_string())
    print(f"\n  Results saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("  ✓ Classical ML training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

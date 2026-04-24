"""
Benchmark Evaluator — Chest X-ray Benchmark
==============================================
Comprehensive evaluation: metrics, confusion matrix, ROC/PR curves,
cross-model comparison, and statistical significance testing.

Classes:
    BenchmarkEvaluator — evaluate and compare all models
Functions:
    plot_confusion_matrix — annotated confusion matrix (2 panels)
    plot_roc_curves — per-class ROC curves
    plot_precision_recall_curves — per-class PR curves
    cross_model_comparison_plot — horizontal bar chart
    statistical_significance_test — Wilcoxon signed-rank test
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)

# Professional plot style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# =============================================================================
# BenchmarkEvaluator
# =============================================================================
class BenchmarkEvaluator:
    """Unified evaluation for all model families.

    Computes and stores: Accuracy, Precision, Recall, F1 (macro/weighted),
    ROC-AUC per class and macro-average.

    Parameters
    ----------
    class_names : list[str]
        List of class label names.
    """

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate a single model.

        Parameters
        ----------
        model_name : str
            Name of the model.
        y_true : np.ndarray
            Ground truth labels, shape (N,).
        y_pred : np.ndarray
            Predicted labels, shape (N,).
        y_proba : np.ndarray, optional
            Predicted probabilities, shape (N, C).

        Returns
        -------
        dict[str, float]
            Dictionary of computed metrics.
        """
        metrics: Dict[str, Any] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }

        # Per-class F1
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, name in enumerate(self.class_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name}"] = round(per_class_f1[i], 4)

        # ROC-AUC (if probabilities available)
        if y_proba is not None:
            try:
                macro_auc = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                metrics["macro_auc"] = round(macro_auc, 4)

                # Per-class AUC
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=list(range(self.n_classes)))
                for i, name in enumerate(self.class_names):
                    if i < y_proba.shape[1]:
                        try:
                            class_auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                            metrics[f"auc_{name}"] = round(class_auc, 4)
                        except ValueError:
                            metrics[f"auc_{name}"] = None
            except Exception as e:
                logger.warning(f"  ROC-AUC failed for {model_name}: {e}")

        self.results[model_name] = metrics
        return metrics

    def comparison_table(self) -> pd.DataFrame:
        """Generate formatted comparison table sorted by Macro F1.

        Returns
        -------
        pd.DataFrame
            Comparison DataFrame with best values highlighted.
        """
        if not self.results:
            return pd.DataFrame()

        # Select primary metrics for comparison
        primary_keys = ["accuracy", "macro_precision", "macro_recall",
                        "macro_f1", "weighted_f1", "macro_auc"]

        rows = []
        for model_name, metrics in self.results.items():
            row = {"Model": model_name}
            for key in primary_keys:
                row[key] = metrics.get(key, None)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        df = df.sort_values("macro_f1", ascending=False)

        return df

    def save_results(self, path: str | Path) -> None:
        """Save all results as JSON.

        Parameters
        ----------
        path : str | Path
            Output JSON file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert None values to null-safe format
        serialisable = {}
        for model, metrics in self.results.items():
            serialisable[model] = {
                k: v if v is not None else "N/A"
                for k, v in metrics.items()
            }

        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"  Results saved: {path}")

    def load_results(self, path: str | Path) -> None:
        """Load previously saved results.

        Parameters
        ----------
        path : str | Path
            JSON file path.
        """
        with open(path) as f:
            self.results = json.load(f)
        logger.info(f"  Results loaded: {path} ({len(self.results)} models)")


# =============================================================================
# Confusion Matrix
# =============================================================================
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str | Path,
) -> None:
    """Plot confusion matrix with raw counts and row-normalised percentages.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predicted labels.
    class_names : list[str]
        Class names.
    model_name : str
        Model name for title.
    save_path : str | Path
        Path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
        xticklabels=class_names, yticklabels=class_names,
    )
    ax1.set_title(f"{model_name} — Counts", fontsize=14)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    # Normalised percentages
    sns.heatmap(
        cm_norm, annot=True, fmt=".1f", cmap="Blues", ax=ax2,
        xticklabels=class_names, yticklabels=class_names,
    )
    ax2.set_title(f"{model_name} — Normalised (%)", fontsize=14)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Confusion matrix saved: {save_path}")


# =============================================================================
# ROC Curves
# =============================================================================
def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str | Path,
) -> None:
    """Plot per-class ROC curves (one-vs-rest).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (N,).
    y_proba : np.ndarray
        Predicted probabilities, shape (N, C).
    class_names : list[str]
        Class names.
    model_name : str
        Model name for title.
    save_path : str | Path
        Save path.
    """
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if i >= y_proba.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} — ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ROC curves saved: {save_path}")


# =============================================================================
# Precision-Recall Curves
# =============================================================================
def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str | Path,
) -> None:
    """Plot per-class precision-recall curves.

    Parameters
    ----------
    y_true, y_proba : np.ndarray
        Labels and probabilities.
    class_names : list[str]
        Class names.
    model_name : str
        Model name.
    save_path : str | Path
        Save path.
    """
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if i >= y_proba.shape[1]:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{model_name} — Precision-Recall Curves", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  PR curves saved: {save_path}")


# =============================================================================
# Cross-Model Comparison
# =============================================================================
def cross_model_comparison_plot(
    results_dict: Dict[str, Dict[str, float]],
    metric: str = "macro_f1",
    save_path: str | Path = "outputs/evaluation/model_comparison.png",
) -> None:
    """Horizontal bar chart comparing all models on a chosen metric.

    Parameters
    ----------
    results_dict : dict
        {model_name: {metric_name: value, ...}, ...}
    metric : str
        Metric to compare. Default 'macro_f1'.
    save_path : str | Path
        Save path.
    """
    # Model family → color mapping
    family_colors = {
        "classical": "#9E9E9E",    # Gray
        "ensemble": "#FFB300",     # Amber
        "cnn": "#1976D2",          # Blue
        "transformer": "#00897B",  # Teal
        "detection": "#E64A19",    # Coral
    }

    def get_family(name: str) -> str:
        name_lower = name.lower()
        if any(k in name_lower for k in ["svm", "logistic"]):
            return "classical"
        if any(k in name_lower for k in ["xgboost", "adaboost"]):
            return "ensemble"
        if any(k in name_lower for k in ["resnet", "efficientnet", "patchlstm"]):
            return "cnn"
        if any(k in name_lower for k in ["swin", "vit", "autoencoder"]):
            return "transformer"
        if any(k in name_lower for k in ["yolo", "faster", "mask"]):
            return "detection"
        return "classical"

    # Sort by metric value
    items = sorted(
        [(name, metrics.get(metric, 0)) for name, metrics in results_dict.items()],
        key=lambda x: x[1],
    )
    names, values = zip(*items) if items else ([], [])

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))

    colors = [family_colors.get(get_family(n), "#9E9E9E") for n in names]
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=f.title()) for f, c in family_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Comparison plot saved: {save_path}")


# =============================================================================
# Statistical Significance
# =============================================================================
def statistical_significance_test(
    results_dict: Dict[str, List[float]],
    method: str = "wilcoxon",
) -> None:
    """Run statistical significance test between top-2 models.

    Parameters
    ----------
    results_dict : dict
        {model_name: list_of_fold_scores, ...}
        Each model should have the same number of fold scores.
    method : str
        Test method: 'wilcoxon'. Default 'wilcoxon'.
    """
    from scipy import stats

    if len(results_dict) < 2:
        print("  Need at least 2 models for significance testing.")
        return

    # Sort by mean score
    sorted_models = sorted(
        results_dict.items(),
        key=lambda x: np.mean(x[1]),
        reverse=True,
    )

    model1_name, scores1 = sorted_models[0]
    model2_name, scores2 = sorted_models[1]

    print(f"\n  Statistical Significance Test ({method})")
    print(f"  {'─' * 50}")
    print(f"  Top-1: {model1_name} (mean={np.mean(scores1):.4f})")
    print(f"  Top-2: {model2_name} (mean={np.mean(scores2):.4f})")

    if method == "wilcoxon":
        try:
            stat, p_value = stats.wilcoxon(scores1, scores2)
            significant = p_value < 0.05

            print(f"\n  Wilcoxon signed-rank test:")
            print(f"    Statistic: {stat:.4f}")
            print(f"    p-value:   {p_value:.6f}")
            print(f"    Significant (p < 0.05): {'YES ✓' if significant else 'NO ✗'}")

            if significant:
                print(f"    → {model1_name} is significantly better than {model2_name}")
            else:
                print(f"    → No statistically significant difference")
        except Exception as e:
            print(f"    Test failed: {e}")
    else:
        print(f"  Unknown method: {method}")

"""
Evaluation Plots — Chest X-ray Benchmark
==========================================
Re-exports from evaluator for cleaner imports.
"""

from src.evaluation.evaluator import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    cross_model_comparison_plot,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "cross_model_comparison_plot",
]

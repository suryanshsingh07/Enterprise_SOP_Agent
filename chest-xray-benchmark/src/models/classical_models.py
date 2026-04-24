"""
Classical ML Models — Chest X-ray Benchmark
=============================================
SVM, Logistic Regression, XGBoost, and AdaBoost classifiers
with Optuna hyperparameter tuning for XGBoost.

Functions:
    build_svm — RBF SVM with probability output
    build_logistic_regression — multinomial LR
    build_xgboost — gradient boosted trees
    build_adaboost — adaptive boosting with decision stumps
    train_classical_model — fit, evaluate, save
    tune_xgboost_optuna — 50-trial Bayesian optimization
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# Model Builders
# =============================================================================

def build_svm(
    C: float = 10.0,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> SVC:
    """Build an SVM classifier for multi-class chest X-ray classification.

    Parameters
    ----------
    C : float
        Regularisation parameter. Tuning range: [0.01, 100]. Default 10.0.
        Higher C → less regularisation → tighter fit to training data.
    kernel : str
        Kernel type. Options: 'rbf', 'linear', 'poly'. Default 'rbf'.
    gamma : str
        Kernel coefficient. Options: 'scale', 'auto'. Default 'scale'.

    Returns
    -------
    SVC
        Configured (unfitted) SVM classifier.
    """
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,           # Needed for ROC-AUC computation
        class_weight="balanced",     # Auto-weight by inverse class frequency
        decision_function_shape="ovr",  # One-vs-rest for multi-class
        random_state=42,
        cache_size=1000,             # 1GB cache for faster training
        verbose=False,
    )


def build_logistic_regression(C: float = 1.0) -> LogisticRegression:
    """Build a Logistic Regression classifier.

    Parameters
    ----------
    C : float
        Inverse regularisation strength. Tuning range: [0.001, 100]. Default 1.0.
        Smaller C → stronger regularisation.

    Returns
    -------
    LogisticRegression
        Configured (unfitted) LR classifier.
    """
    return LogisticRegression(
        C=C,
        solver="lbfgs",             # Good for multi-class + large datasets
        max_iter=2000,               # Ensure convergence
        multi_class="multinomial",   # True multi-class (not OvR)
        class_weight="balanced",     # Handle class imbalance
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost(n_classes: int = 6) -> "XGBClassifier":
    """Build an XGBoost classifier.

    Parameters
    ----------
    n_classes : int
        Number of output classes. Default 6.

    Returns
    -------
    XGBClassifier
        Configured (unfitted) XGBoost classifier.

    Notes
    -----
    Hyperparameter tuning ranges (for Optuna):
    - n_estimators: [100, 500]
    - max_depth: [3, 8]
    - learning_rate: [0.01, 0.3] (log-uniform)
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.6, 1.0]
    - min_child_weight: [1, 10]
    - gamma: [0, 5]
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,            # Range: [100, 500]. More trees = more capacity
        max_depth=6,                 # Range: [3, 8]. Deeper = more complex splits
        learning_rate=0.05,          # Range: [0.01, 0.3]. Lower = needs more trees
        subsample=0.8,               # Range: [0.6, 1.0]. Row sampling fraction
        colsample_bytree=0.8,       # Range: [0.6, 1.0]. Feature sampling per tree
        min_child_weight=3,          # Range: [1, 10]. Min samples to create a leaf
        gamma=0.1,                   # Range: [0, 5]. Min loss reduction for split
        objective="multi:softprob",  # Multi-class probability output
        num_class=n_classes,
        tree_method="hist",          # Fast histogram-based method
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )


def build_adaboost() -> AdaBoostClassifier:
    """Build an AdaBoost classifier with decision stumps.

    Uses shallow decision trees (max_depth=2) as base estimator.

    Returns
    -------
    AdaBoostClassifier
        Configured (unfitted) AdaBoost classifier.

    Notes
    -----
    Hyperparameter tuning ranges:
    - n_estimators: [50, 500]
    - learning_rate: [0.01, 1.0]
    - base max_depth: [1, 5]
    """
    base_estimator = DecisionTreeClassifier(
        max_depth=2,                # Weak learner (stump-like)
        random_state=42,
    )
    return AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=200,            # Range: [50, 500]
        learning_rate=0.1,           # Range: [0.01, 1.0]. Lower = more robust
        algorithm="SAMME",
        random_state=42,
    )


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_classical_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    save_dir: str | Path,
    class_names: Optional[list[str]] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train a classical ML model, evaluate on validation set, and save.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Unfitted model.
    X_train : np.ndarray
        Training features, shape (N_train, D).
    y_train : np.ndarray
        Training labels, shape (N_train,).
    X_val : np.ndarray
        Validation features, shape (N_val, D).
    y_val : np.ndarray
        Validation labels, shape (N_val,).
    model_name : str
        Name for logging and file naming.
    save_dir : str | Path
        Directory to save the fitted model.
    class_names : list[str], optional
        Class label names for the classification report.

    Returns
    -------
    tuple[Any, dict[str, float]]
        (fitted_model, metrics_dict)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        from src.data.dataset_builder import CLASS_NAMES
        class_names = CLASS_NAMES

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples")

    # Fit
    model.fit(X_train, y_train)
    print(f"  ✓ Model fitted.")

    # Predict
    y_pred = model.predict(X_val)

    # Probabilities (if available)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    macro_p = precision_score(y_val, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_val, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    metrics: Dict[str, float] = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }

    # ROC-AUC (if probabilities available)
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")
            metrics["macro_auc"] = round(auc, 4)
        except Exception as e:
            logger.warning(f"  ROC-AUC computation failed: {e}")

    # Classification report
    print(f"\n  Classification Report ({model_name}):")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

    # Save model
    model_path = save_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    # Save metrics
    metrics_path = save_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


# =============================================================================
# Optuna Hyperparameter Tuning for XGBoost
# =============================================================================

def tune_xgboost_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    n_classes: int = 6,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using Optuna with macro F1 objective.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (N_train, D).
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation labels.
    n_trials : int
        Number of Optuna trials. Default 50.
    n_classes : int
        Number of output classes. Default 6.

    Returns
    -------
    dict[str, Any]
        Best hyperparameters found.
    """
    import optuna
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        model = XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"\n  Running Optuna XGBoost tuning ({n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_chest_xray",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_f1 = study.best_value

    print(f"\n  Best Macro F1: {best_f1:.4f}")
    print(f"  Best Params:")
    for k, v in best.items():
        print(f"    {k}: {v}")

    return best


if __name__ == "__main__":
    print("=" * 60)
    print("  Classical Models — Demo")
    print("=" * 60)

    # Dummy data
    rng = np.random.RandomState(42)
    X = rng.randn(200, 512).astype(np.float32)
    y = rng.randint(0, 6, 200)

    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]

    # Build all models
    models = {
        "SVM": build_svm(),
        "LogisticRegression": build_logistic_regression(),
        "XGBoost": build_xgboost(),
        "AdaBoost": build_adaboost(),
    }

    results = {}
    for name, model in models.items():
        _, metrics = train_classical_model(
            model, X_train, y_train, X_val, y_val,
            model_name=name,
            save_dir="outputs/classical",
        )
        results[name] = metrics

    # Print comparison
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:25s} │ F1={m['macro_f1']:.4f} │ Acc={m['accuracy']:.4f}")

    print("\n  ✓ Classical models demo complete.")

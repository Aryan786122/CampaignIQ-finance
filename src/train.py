"""Model training, validation comparison, and pipeline selection for CampaignIQ.

Implements:
- Model A: Logistic Regression (interpretable linear baseline)
- Model B: Gradient Boosting (HistGradientBoostingClassifier)

Validation comparison uses Average Precision (PR-AUC) and calibration metrics,
avoiding simple accuracy on this imbalanced dataset.
"""

from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor


def train_logistic_regression(random_state: int = 42) -> Pipeline:
    """Create a full pipeline with Logistic Regression."""
    preprocessor = create_preprocessor()
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_gradient_boosting(random_state: int = 42) -> Pipeline:
    """Create a full pipeline with HistGradientBoostingClassifier."""
    preprocessor = create_preprocessor()
    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=25,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def find_optimal_threshold(y_val: np.ndarray, y_val_scores: np.ndarray) -> float:
    """Find decision threshold that maximizes F1 score on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_scores)
    # Avoid division by zero
    f1_scores = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        p, r = precisions[i], recalls[i]
        f1_scores[i] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return best_threshold


def evaluate_on_validation(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Calculate validation metrics to compare candidate models."""
    y_scores = model.predict_proba(X_val)[:, 1]

    pr_auc = float(average_precision_score(y_val, y_scores))
    roc_auc = float(roc_auc_score(y_val, y_scores))
    brier = float(brier_score_loss(y_val, y_scores))

    best_thresh = find_optimal_threshold(y_val.to_numpy(), y_scores)
    y_pred_optimal = (y_scores >= best_thresh).astype(int)
    val_f1 = float(f1_score(y_val, y_pred_optimal))

    return {
        "val_average_precision": pr_auc,
        "val_roc_auc": roc_auc,
        "val_brier_score": brier,
        "val_optimal_threshold": best_thresh,
        "val_optimal_f1": val_f1,
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> tuple[str, Pipeline, dict[str, Any]]:
    """Train both models on training set, evaluate on validation set, and select champion."""
    print("Training Model A: Logistic Regression baseline...")
    log_reg_pipeline = train_logistic_regression(random_state=random_state)
    log_reg_pipeline.fit(X_train, y_train)
    log_reg_val_metrics = evaluate_on_validation(log_reg_pipeline, X_val, y_val)
    print(
        f"Model A (Logistic Regression) Val PR-AUC: {log_reg_val_metrics['val_average_precision']:.4f}, "
        f"ROC-AUC: {log_reg_val_metrics['val_roc_auc']:.4f}, "
        f"Brier: {log_reg_val_metrics['val_brier_score']:.4f}"
    )

    print("Training Model B: HistGradientBoosting...")
    gb_pipeline = train_gradient_boosting(random_state=random_state)
    gb_pipeline.fit(X_train, y_train)
    gb_val_metrics = evaluate_on_validation(gb_pipeline, X_val, y_val)
    print(
        f"Model B (Gradient Boosting) Val PR-AUC: {gb_val_metrics['val_average_precision']:.4f}, "
        f"ROC-AUC: {gb_val_metrics['val_roc_auc']:.4f}, "
        f"Brier: {gb_val_metrics['val_brier_score']:.4f}"
    )

    # Model selection based on validation PR-AUC (ranking performance on imbalanced positive class)
    # and calibration (lower Brier score)
    if gb_val_metrics["val_average_precision"] >= log_reg_val_metrics["val_average_precision"]:
        selected_name = "HistGradientBoosting"
        champion_pipeline = gb_pipeline
        rationale = (
            f"HistGradientBoosting achieved higher validation Average Precision "
            f"({gb_val_metrics['val_average_precision']:.4f} vs {log_reg_val_metrics['val_average_precision']:.4f}) "
            f"and superior ranking discrimination on non-linear interaction features."
        )
    else:
        selected_name = "LogisticRegression"
        champion_pipeline = log_reg_pipeline
        rationale = (
            f"LogisticRegression achieved higher validation Average Precision "
            f"({log_reg_val_metrics['val_average_precision']:.4f} vs {gb_val_metrics['val_average_precision']:.4f}) "
            f"with strong linear baseline interpretability."
        )

    comparison_results = {
        "selected_model": selected_name,
        "selection_rationale": rationale,
        "logistic_regression": log_reg_val_metrics,
        "gradient_boosting": gb_val_metrics,
        "chosen_threshold": (
            gb_val_metrics["val_optimal_threshold"]
            if selected_name == "HistGradientBoosting"
            else log_reg_val_metrics["val_optimal_threshold"]
        ),
    }

    return selected_name, champion_pipeline, comparison_results

"""Evaluation module for CampaignIQ models on held-out test data.

Calculates comprehensive ranking, threshold-based classification, and calibration metrics.
Outputs evaluation artifacts.
"""

import json
import os
from typing import Any
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

from src.decision_engine import evaluate_outreach


def evaluate_model_on_test(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float,
    outreach_budget: float = 0.20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute complete test evaluation metrics and generate test prediction DataFrame."""
    y_true = y_test.to_numpy()
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= decision_threshold).astype(int)

    # Core ranking metrics (threshold-independent)
    pr_auc = float(average_precision_score(y_true, y_scores))
    roc_auc = float(roc_auc_score(y_true, y_scores))
    brier = float(brier_score_loss(y_true, y_scores))

    # Threshold-dependent metrics (using validation-selected threshold)
    test_accuracy = float(accuracy_score(y_true, y_pred))
    test_precision = float(precision_score(y_true, y_pred, zero_division=0))
    test_recall = float(recall_score(y_true, y_pred, zero_division=0))
    test_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Baseline majority class accuracy
    neg_count = int(np.sum(y_true == 0))
    pos_count = int(np.sum(y_true == 1))
    naive_majority_accuracy = float(neg_count / len(y_true))
    baseline_prevalence = float(pos_count / len(y_true))

    # Decision Engine metrics across budget spectrum
    budget_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    decision_metrics = {}
    for r in budget_ratios:
        res = evaluate_outreach(y_true, y_scores, outreach_ratio=r)
        pct_key = f"{int(round(r * 100))}%"
        decision_metrics[pct_key] = res.to_dict()

    # Calibration curve diagnostics (5 bins)
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=5, strategy="quantile")
    calibration_diagnostics = {
        "brier_score": brier,
        "quantile_bins": {
            "mean_predicted_probabilities": [float(p) for p in prob_pred],
            "empirical_positive_rates": [float(p) for p in prob_true],
        },
    }

    metrics = {
        "dataset_summary": {
            "total_test_samples": len(y_true),
            "test_positive_count": pos_count,
            "test_negative_count": neg_count,
            "test_baseline_prevalence": baseline_prevalence,
            "naive_majority_class_accuracy": naive_majority_accuracy,
        },
        "ranking_metrics": {
            "average_precision_pr_auc": pr_auc,
            "roc_auc": roc_auc,
        },
        "classification_metrics": {
            "decision_threshold": decision_threshold,
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "confusion_matrix": cm,
            "accuracy_warning": (
                f"Test dataset has a baseline positive rate of {baseline_prevalence:.1%}. "
                f"A naive classifier predicting 0 for all customers yields {naive_majority_accuracy:.1%} accuracy "
                f"while finding 0 interested customers. Hence PR-AUC and Lift-at-K are the primary decision metrics."
            ),
        },
        "outreach_decision_metrics": decision_metrics,
        "primary_outreach_result": decision_metrics[f"{int(round(outreach_budget * 100))}%"],
        "calibration": calibration_diagnostics,
    }

    # Prediction dataframe
    preds_df = pd.DataFrame(
        {
            "original_index": X_test.index,
            "y_true": y_true,
            "y_score": y_scores,
            "y_pred": y_pred,
        }
    )

    return metrics, preds_df


def save_evaluation_artifacts(
    metrics: dict[str, Any],
    preds_df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> None:
    """Save metrics.json and test_predictions.csv to artifacts directory."""
    os.makedirs(artifacts_dir, exist_ok=True)

    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    preds_path = os.path.join(artifacts_dir, "test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved test predictions to {preds_path} ({len(preds_df):,} rows)")

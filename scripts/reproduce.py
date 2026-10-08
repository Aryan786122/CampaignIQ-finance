"""Reproduce all CampaignIQ experiments end-to-end.

Workflow:
1. Downloads official UCI Bank Marketing dataset (41,188 records).
2. Performs chronological split (70% train, 15% validation, 15% test).
3. Preprocesses data with zero leakage (drops duration & campaign; fits only on train).
4. Trains Model A (Logistic Regression) and Model B (HistGradientBoosting).
5. Compares validation metrics and selects champion model.
6. Evaluates champion model on held-out test set.
7. Computes validation permutation feature importance.
8. Saves all artifacts: metrics.json, test_predictions.csv, feature_importance.csv, model.joblib.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from src.data import (
    download_bank_data,
    load_raw_data,
    prepare_features,
    chronological_split,
)
from src.train import train_and_compare_models
from src.evaluate import evaluate_model_on_test, save_evaluation_artifacts
from src.explain import compute_feature_importance, analyze_failure_modes


def main() -> None:
    print("=" * 70)
    print(" CampaignIQ — Reproducible Training & Evaluation Pipeline")
    print("=" * 70)

    # 1. Download & verify dataset
    csv_path = download_bank_data("data/raw")
    raw_df = load_raw_data(csv_path)
    print(f"Loaded raw dataset: {len(raw_df):,} rows x {raw_df.shape[1]} columns.")

    # 2. Chronological Split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = chronological_split(raw_df, 0.70, 0.15, 0.15)
    print(
        f"Chronological split: Train={len(train_df):,} (70%), "
        f"Val={len(val_df):,} (15%), Test={len(test_df):,} (15%)"
    )

    # 3. Leakage-safe feature extraction
    X_train, y_train = prepare_features(train_df, drop_leakage=True)
    X_val, y_val = prepare_features(val_df, drop_leakage=True)
    X_test, y_test = prepare_features(test_df, drop_leakage=True)

    print(f"Features prepared: {X_train.shape[1]} features (duration & campaign excluded).")
    print(f"Train positive rate: {y_train.mean():.2%}")
    print(f"Val positive rate:   {y_val.mean():.2%}")
    print(f"Test positive rate:  {y_test.mean():.2%}")

    # 4. Model Training & Validation Comparison
    selected_name, champion_model, comparison = train_and_compare_models(
        X_train, y_train, X_val, y_val, random_state=42
    )

    # Save champion model pipeline
    os.makedirs("artifacts", exist_ok=True)
    model_artifact_path = os.path.join("artifacts", "model.joblib")
    joblib.dump(champion_model, model_artifact_path)
    print(f"Saved champion pipeline ({selected_name}) to {model_artifact_path}")

    # 5. Evaluate Champion on Held-out Test Set
    optimal_threshold = comparison["chosen_threshold"]
    print(f"\nEvaluating on held-out test set using validation threshold {optimal_threshold:.4f}...")
    metrics, preds_df = evaluate_model_on_test(
        model=champion_model,
        X_test=X_test,
        y_test=y_test,
        decision_threshold=optimal_threshold,
        outreach_budget=0.20,
    )

    # Add model comparison details to metrics
    metrics["model_selection"] = comparison

    # 6. Permutation Feature Importance (Validation set to avoid test snooping)
    fi_df = compute_feature_importance(
        model=champion_model,
        X_val=X_val,
        y_val=y_val,
        n_repeats=10,
        random_state=42,
        artifacts_dir="artifacts",
    )

    # 7. Failure mode diagnostics
    failure_diag = analyze_failure_modes(preds_df, X_test, top_n=5)
    metrics["failure_analysis"] = failure_diag

    # 8. Save artifacts
    save_evaluation_artifacts(metrics, preds_df, artifacts_dir="artifacts")

    # 9. Summary Print
    print("\n" + "=" * 70)
    print(" EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"Champion Model:       {selected_name}")
    print(f"Held-out PR-AUC:      {metrics['ranking_metrics']['average_precision_pr_auc']:.4f}")
    print(f"Held-out ROC-AUC:     {metrics['ranking_metrics']['roc_auc']:.4f}")
    print(f"Naive Majority Acc:   {metrics['dataset_summary']['naive_majority_class_accuracy']:.2%}")
    print(f"Balanced Threshold:   {optimal_threshold:.4f}")
    print(f"Held-out Precision:   {metrics['classification_metrics']['precision']:.4f}")
    print(f"Held-out Recall:      {metrics['classification_metrics']['recall']:.4f}")
    print(f"Held-out F1 Score:    {metrics['classification_metrics']['f1_score']:.4f}")
    print(f"Brier Score:          {metrics['calibration']['brier_score']:.4f}")

    p20 = metrics["primary_outreach_result"]
    print("\n--- Decision Engine (20% Outreach Budget) ---")
    print(f"Customers Selected:   {p20['k_selected']:,} of {p20['total_customers']:,}")
    print(f"Observed Positives:   {p20['observed_positives']:,}")
    print(f"Expected Random Pos:  {p20['expected_random_positives']:.1f}")
    print(f"Precision-at-20%:     {p20['precision_at_k']:.2%}")
    print(f"Lift-at-20%:          {p20['lift_at_k']:.2f}x")
    print(f"Positives Captured:   {p20['positive_capture_rate']:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()

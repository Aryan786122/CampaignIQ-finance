"""Feature explainability and error diagnostics for CampaignIQ.

Computes global permutation feature importance on the validation set
and analyzes model failure modes on test predictions.
"""

import os
from typing import Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def compute_feature_importance(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    artifacts_dir: str = "artifacts",
) -> pd.DataFrame:
    """Compute permutation importance on the validation set using average precision scoring.

    Permuting features in X_val measures each original pre-contact attribute's contribution
    to ranking performance without snooping on the test set.
    """
    print("Computing validation permutation feature importance (reproducible seed 42)...")
    # Subsample if validation set is very large to balance precision and speed
    val_sample_size = min(len(X_val), 3000)
    if len(X_val) > val_sample_size:
        X_sample = X_val.sample(n=val_sample_size, random_state=random_state)
        y_sample = y_val.loc[X_sample.index]
    else:
        X_sample = X_val
        y_sample = y_val

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    fi_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    os.makedirs(artifacts_dir, exist_ok=True)
    out_csv = os.path.join(artifacts_dir, "feature_importance.csv")
    fi_df.to_csv(out_csv, index=False)
    print(f"Saved feature importances to {out_csv}")
    print("Top 5 features by validation permutation importance:")
    for idx, row in fi_df.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

    return fi_df


def analyze_failure_modes(
    preds_df: pd.DataFrame,
    X_test: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """Examine false positives and false negatives to diagnose failure patterns.

    - False Positives: Customers with highest predicted probability who did NOT subscribe.
    - False Negatives: Customers who subscribed despite having the lowest predicted probability.
    """
    merged = preds_df.copy()
    for col in ["euribor3m", "poutcome", "age", "job", "education"]:
        if col in X_test.columns:
            merged[col] = X_test[col].values

    # High-confidence False Positives (y_true == 0, highest scores)
    fps = (
        merged[merged["y_true"] == 0]
        .sort_values("y_score", ascending=False)
        .head(top_n)
    )

    # Low-confidence False Negatives (y_true == 1, lowest scores)
    fns = (
        merged[merged["y_true"] == 1]
        .sort_values("y_score", ascending=True)
        .head(top_n)
    )

    return {
        "false_positives_sample": fps[["y_score", "euribor3m", "poutcome", "age", "job"]].to_dict(orient="records"),
        "false_negatives_sample": fns[["y_score", "euribor3m", "poutcome", "age", "job"]].to_dict(orient="records"),
    }

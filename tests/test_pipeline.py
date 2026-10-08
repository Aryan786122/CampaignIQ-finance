"""Tests for preprocessing, model pipeline inference, artifact loading, and validation."""

import os
import json
import joblib
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.train import train_logistic_regression, train_gradient_boosting


@pytest.fixture
def sample_feature_df():
    data = {
        "age": [32, 54, 21, 40],
        "pdays": [999, 3, 999, 999],
        "previous": [0, 1, 0, 0],
        "emp.var.rate": [1.1, -1.8, 1.4, -0.1],
        "cons.price.idx": [93.994, 92.893, 94.465, 93.200],
        "cons.conf.idx": [-36.4, -46.2, -41.8, -42.0],
        "euribor3m": [4.857, 1.281, 4.962, 4.021],
        "nr.employed": [5191.0, 5099.1, 5228.1, 5195.8],
        "job": ["admin.", "services", "student", "technician"],
        "marital": ["single", "married", "single", "divorced"],
        "education": ["university.degree", "high.school", "basic.4y", "professional.course"],
        "default": ["no", "no", "no", "unknown"],
        "housing": ["yes", "no", "yes", "yes"],
        "loan": ["no", "no", "no", "no"],
        "contact": ["cellular", "cellular", "telephone", "cellular"],
        "month": ["may", "nov", "aug", "jul"],
        "day_of_week": ["mon", "fri", "tue", "thu"],
        "poutcome": ["nonexistent", "success", "nonexistent", "failure"],
    }
    return pd.DataFrame(data)


def test_preprocessor_transformation(sample_feature_df):
    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(sample_feature_df)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_feature_df)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_categories(sample_feature_df):
    preprocessor = create_preprocessor()
    preprocessor.fit(sample_feature_df)

    unseen_df = sample_feature_df.copy()
    unseen_df.loc[0, "job"] = "astronaut"  # Category never seen in fit
    unseen_df.loc[0, "marital"] = "novel_status"

    # Must transform without error (handle_unknown='ignore')
    transformed = preprocessor.transform(unseen_df)
    assert transformed.shape[0] == len(unseen_df)
    assert not np.isnan(transformed).any()


def test_model_pipeline_predict_proba(sample_feature_df):
    y = pd.Series([0, 1, 0, 1])
    pipeline = train_logistic_regression()
    pipeline.fit(sample_feature_df, y)

    probs = pipeline.predict_proba(sample_feature_df)
    assert probs.shape == (4, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)


def test_saved_artifacts_exist_and_load():
    model_path = "artifacts/model.joblib"
    metrics_path = "artifacts/metrics.json"
    preds_path = "artifacts/test_predictions.csv"
    fi_path = "artifacts/feature_importance.csv"

    assert os.path.exists(model_path), f"Missing {model_path}"
    assert os.path.exists(metrics_path), f"Missing {metrics_path}"
    assert os.path.exists(preds_path), f"Missing {preds_path}"
    assert os.path.exists(fi_path), f"Missing {fi_path}"

    # Load model artifact
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict_proba")

    # Load metrics artifact
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "ranking_metrics" in metrics
    assert "average_precision_pr_auc" in metrics["ranking_metrics"]
    assert "primary_outreach_result" in metrics
    assert "outreach_decision_metrics" in metrics

    # Load predictions artifact
    preds_df = pd.read_csv(preds_path)
    assert len(preds_df) == 6179
    assert {"y_true", "y_score", "y_pred"}.issubset(preds_df.columns)

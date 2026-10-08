"""Unit tests for CampaignIQ Decision Intelligence Engine."""

import pytest
import numpy as np
import pandas as pd
from src.decision_engine import (
    evaluate_outreach,
    rank_and_select,
    validate_inputs,
    compute_cumulative_gains,
)


def test_validate_inputs_valid():
    y_true = [0, 1, 0, 1]
    y_scores = [0.1, 0.8, 0.2, 0.9]
    yt, ys = validate_inputs(y_true, y_scores, 0.5)
    assert len(yt) == 4
    assert len(ys) == 4


def test_validate_inputs_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_inputs([], [], 0.2)


def test_validate_inputs_mismatched_length():
    with pytest.raises(ValueError, match="Length mismatch"):
        validate_inputs([0, 1], [0.5], 0.2)


def test_validate_inputs_invalid_ratio():
    with pytest.raises(ValueError, match="outreach_ratio must be strictly between"):
        validate_inputs([0, 1], [0.2, 0.8], 0.0)
    with pytest.raises(ValueError, match="outreach_ratio must be strictly between"):
        validate_inputs([0, 1], [0.2, 0.8], 1.5)


def test_validate_inputs_non_binary_labels():
    with pytest.raises(ValueError, match="binary values"):
        validate_inputs([0, 2], [0.2, 0.8], 0.5)


def test_rank_and_select_ordering():
    # 5 samples: indices 0, 1, 2, 3, 4
    y_true = np.array([0, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.3, 0.7, 0.2])
    # Top 40% means k = round(5 * 0.4) = 2
    k, top_labels, top_scores = rank_and_select(y_true, y_scores, 0.4)
    assert k == 2
    # Highest scores are index 1 (0.9, label 1) and index 3 (0.7, label 1)
    np.testing.assert_array_equal(top_labels, [1, 1])
    np.testing.assert_array_equal(top_scores, [0.9, 0.7])


def test_rank_and_select_stable_with_ties():
    # Ties in scores should not crash or produce invalid lengths
    y_true = np.array([1, 0, 1, 0])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5])
    k, top_labels, top_scores = rank_and_select(y_true, y_scores, 0.5)
    assert k == 2
    assert len(top_labels) == 2


def test_evaluate_outreach_calculation():
    # 10 samples: 2 positives, 8 negatives. Baseline prevalence = 2/10 = 0.20 (20%)
    # Perfect ranking puts 2 positives at the very top
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_scores = np.array([0.95, 0.85, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])

    # Test 20% outreach budget -> K = 2
    res = evaluate_outreach(y_true, y_scores, outreach_ratio=0.20)

    assert res.total_customers == 10
    assert res.k_selected == 2
    assert res.observed_positives == 2
    assert res.baseline_prevalence == pytest.approx(0.20)
    assert res.precision_at_k == pytest.approx(1.0)
    assert res.expected_random_positives == pytest.approx(2 * 0.20)  # 0.4
    assert res.lift_at_k == pytest.approx(1.0 / 0.20)  # 5.0x
    assert res.positive_capture_rate == pytest.approx(1.0)  # Captured 100% of positives
    assert "5.00x lift" in res.interpretation


def test_evaluate_outreach_different_percentages():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])  # 3 positives, 10 samples
    y_scores = np.linspace(0.1, 1.0, 10)  # Ascending, so highest score is index 9

    # Test 10%
    r10 = evaluate_outreach(y_true, y_scores, outreach_ratio=0.10)
    assert r10.k_selected == 1

    # Test 50%
    r50 = evaluate_outreach(y_true, y_scores, outreach_ratio=0.50)
    assert r50.k_selected == 5

    # Test 100%
    r100 = evaluate_outreach(y_true, y_scores, outreach_ratio=1.0)
    assert r100.k_selected == 10
    assert r100.lift_at_k == pytest.approx(1.0)


def test_evaluate_outreach_zero_positives():
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4])
    res = evaluate_outreach(y_true, y_scores, outreach_ratio=0.5)
    assert res.baseline_prevalence == 0.0
    assert res.lift_at_k == 0.0
    assert res.precision_at_k == 0.0
    assert res.expected_random_positives == 0.0


def test_compute_cumulative_gains():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    df_gains = compute_cumulative_gains(y_true, y_scores, steps=5)
    assert isinstance(df_gains, pd.DataFrame)
    assert len(df_gains) == 5
    assert "lift_at_k" in df_gains.columns
    assert "expected_random_positives" in df_gains.columns

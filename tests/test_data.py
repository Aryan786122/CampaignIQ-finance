"""Tests for data ingestion, schema validation, leakage prevention, and chronological splitting."""

import os
import pytest
import pandas as pd
import numpy as np
from src.data import (
    load_raw_data,
    prepare_features,
    chronological_split,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)

DATA_PATH = "data/raw/bank-additional-full.csv"


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Dataset file not downloaded yet")
def test_load_raw_data_shape_and_columns():
    df = load_raw_data(DATA_PATH)
    assert len(df) == 41188
    assert TARGET_COLUMN in df.columns
    for col in LEAKAGE_COLUMNS:
        assert col in df.columns


def test_prepare_features_drops_leakage():
    # Synthetic test DataFrame with leakage columns and target
    synthetic_df = pd.DataFrame(
        {
            "age": [30, 45, 22],
            "job": ["admin.", "services", "student"],
            "duration": [120, 450, 60],  # Leakage
            "campaign": [1, 2, 1],       # Leakage
            "euribor3m": [4.85, 4.85, 1.20],
            "y": ["no", "yes", "no"],
        }
    )

    X, y = prepare_features(synthetic_df, drop_leakage=True)

    assert "duration" not in X.columns
    assert "campaign" not in X.columns
    assert "y" not in X.columns
    assert "age" in X.columns
    assert "euribor3m" in X.columns
    np.testing.assert_array_equal(y.to_numpy(), [0, 1, 0])


def test_chronological_split_ratios_and_order():
    n_samples = 100
    dummy_df = pd.DataFrame(
        {
            "idx": np.arange(n_samples),
            "val": np.linspace(0, 100, n_samples),
            "y": ["no"] * 90 + ["yes"] * 10,
        }
    )

    train_df, val_df, test_df = chronological_split(dummy_df, 0.70, 0.15, 0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert len(train_df) + len(val_df) + len(test_df) == n_samples

    # Verify chronological continuity (no shuffling)
    assert train_df["idx"].iloc[-1] == 69
    assert val_df["idx"].iloc[0] == 70
    assert val_df["idx"].iloc[-1] == 84
    assert test_df["idx"].iloc[0] == 85
    assert test_df["idx"].iloc[-1] == 99


def test_chronological_split_invalid_ratios():
    dummy_df = pd.DataFrame({"y": [0, 1]})
    with pytest.raises(AssertionError, match="must sum to 1.0"):
        chronological_split(dummy_df, 0.5, 0.2, 0.2)

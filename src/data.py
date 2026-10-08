"""Data acquisition, validation, and chronological splitting for CampaignIQ.

Dataset: UCI Machine Learning Repository - Bank Marketing
URL: https://archive.ics.uci.edu/dataset/222/bank+marketing
File: bank-additional-full.csv (41,188 records, 20 features + 1 target)
"""

import io
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

UCI_BANK_MARKETING_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ["duration", "campaign"]


def download_bank_data(dest_dir: str = "data/raw") -> str:
    """Download the official UCI Bank Marketing dataset if not already present.

    Extracts `bank-additional-full.csv` into `dest_dir`.
    Returns the absolute path to `bank-additional-full.csv`.
    """
    os.makedirs(dest_dir, exist_ok=True)
    target_csv = os.path.join(dest_dir, "bank-additional-full.csv")

    if os.path.exists(target_csv) and os.path.getsize(target_csv) > 0:
        return os.path.abspath(target_csv)

    print(f"Downloading official dataset from {UCI_BANK_MARKETING_ZIP_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(UCI_BANK_MARKETING_ZIP_URL, headers=headers)

    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    # Outer zip contains bank.zip and bank-additional.zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer_zip:
        if "bank-additional.zip" not in outer_zip.namelist():
            raise FileNotFoundError("bank-additional.zip not found inside root zip archive.")
        sub_zip_bytes = outer_zip.read("bank-additional.zip")

    # Inner zip contains bank-additional/bank-additional-full.csv
    with zipfile.ZipFile(io.BytesIO(sub_zip_bytes)) as inner_zip:
        csv_filename = "bank-additional/bank-additional-full.csv"
        if csv_filename not in inner_zip.namelist():
            raise FileNotFoundError(f"{csv_filename} not found inside bank-additional.zip")
        csv_content = inner_zip.read(csv_filename)

    with open(target_csv, "wb") as f:
        f.write(csv_content)

    print(f"Saved {target_csv} ({len(csv_content):,} bytes).")
    return os.path.abspath(target_csv)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw semicolon-separated CSV dataset and validate schema.

    Expected: 41,188 rows, semicolon separated.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath, sep=";", quotechar='"')
    if len(df) != 41188:
        raise ValueError(f"Expected 41,188 observations in bank-additional-full.csv, found {len(df)}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {df.columns.tolist()}")

    return df


def prepare_features(df: pd.DataFrame, drop_leakage: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from raw DataFrame.

    Converts target 'y' ('yes'/'no') to binary integers (1/0).
    Optionally drops leakage features ('duration', 'campaign').
    """
    df = df.copy()
    y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    if drop_leakage:
        # Exclude duration (unavailable before call) and campaign (includes current call)
        X = X.drop(columns=[col for col in LEAKAGE_COLUMNS if col in X.columns])

    return X, y


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically (train 70%, val 15%, test 15%).

    The bank-additional dataset is ordered chronologically by contact date
    (May 2008 to November 2010). Chronological splitting respects the temporal
    nature of campaign data and avoids look-ahead bias.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    n = len(df)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df

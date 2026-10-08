"""Decision Intelligence Engine for CampaignIQ.

Calculates outreach prioritization metrics comparing ML ranking against
a uniform random targeting baseline expectation.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutreachResult:
    """Summary of outreach decision metrics for a specified budget."""

    total_customers: int
    outreach_ratio: float
    k_selected: int
    observed_positives: int
    precision_at_k: float
    baseline_prevalence: float
    expected_random_positives: float
    lift_at_k: float
    positive_capture_rate: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a plain dictionary."""
        return {
            "total_customers": self.total_customers,
            "outreach_ratio": self.outreach_ratio,
            "k_selected": self.k_selected,
            "observed_positives": self.observed_positives,
            "precision_at_k": self.precision_at_k,
            "baseline_prevalence": self.baseline_prevalence,
            "expected_random_positives": self.expected_random_positives,
            "lift_at_k": self.lift_at_k,
            "positive_capture_rate": self.positive_capture_rate,
            "interpretation": self.interpretation,
        }


def validate_inputs(y_true: np.ndarray, y_scores: np.ndarray, outreach_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Validate and sanitize inputs for the decision engine."""
    y_true_arr = np.asarray(y_true).ravel()
    y_scores_arr = np.asarray(y_scores).ravel()

    if len(y_true_arr) == 0:
        raise ValueError("y_true and y_scores must not be empty.")

    if len(y_true_arr) != len(y_scores_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} items, y_scores has {len(y_scores_arr)} items."
        )

    if not (0.0 < outreach_ratio <= 1.0):
        raise ValueError(f"outreach_ratio must be strictly between 0.0 and 1.0, got {outreach_ratio}.")

    # Validate binary labels
    unique_vals = set(np.unique(y_true_arr))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"y_true must contain binary values (0 or 1), found: {unique_vals}")

    return y_true_arr, y_scores_arr


def rank_and_select(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    outreach_ratio: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Rank observations by predicted probability descending and select top K.

    Uses mergesort (a stable sort) so identical scores preserve determinism.
    Returns:
        k: Number of customers selected
        top_k_labels: Binary outcomes of the top K selected
        top_k_scores: Prediction scores of the top K selected
    """
    y_true_arr, y_scores_arr = validate_inputs(y_true, y_scores, outreach_ratio)
    n = len(y_true_arr)

    # K = integer number of observations selected for contact
    k = int(round(n * outreach_ratio))
    # Bound K between 1 and n
    k = max(1, min(k, n))

    # Stable descending sort by score
    sorted_indices = np.argsort(-y_scores_arr, kind="mergesort")
    top_k_indices = sorted_indices[:k]

    return k, y_true_arr[top_k_indices], y_scores_arr[top_k_indices]


def evaluate_outreach(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    outreach_ratio: float = 0.20,
) -> OutreachResult:
    """Evaluate decision intelligence performance for a specified outreach budget.

    Definitions:
    - K = number of observations selected for contact.
    - Precision-at-K = observed positives among selected observations divided by K.
    - Random baseline expectation = K multiplied by held-out response prevalence.
    - Lift-at-K = precision-at-K divided by held-out response prevalence.
    - Positive capture rate (Recall-at-K) = observed positives in top K divided by total positives.

    Note: Random baseline is explicitly an expectation under uniform random sampling,
    not the measured outcome of an actual randomized experiment.
    """
    y_true_arr, y_scores_arr = validate_inputs(y_true, y_scores, outreach_ratio)
    n = len(y_true_arr)
    total_positives = int(np.sum(y_true_arr))
    baseline_prevalence = float(total_positives / n)

    k, top_k_labels, _ = rank_and_select(y_true_arr, y_scores_arr, outreach_ratio)
    observed_positives = int(np.sum(top_k_labels))

    precision_at_k = float(observed_positives / k)
    expected_random_positives = float(k * baseline_prevalence)

    # Handle zero baseline prevalence safely
    if baseline_prevalence > 0.0:
        lift_at_k = float(precision_at_k / baseline_prevalence)
        capture_rate = float(observed_positives / total_positives)
    else:
        lift_at_k = 0.0
        capture_rate = 0.0

    pct_str = f"{int(round(outreach_ratio * 100))}%"
    interpretation = (
        f"At a {pct_str} outreach budget ({k:,} customers selected out of {n:,}), "
        f"the model-prioritized list contains {observed_positives:,} historical positive responses, "
        f"compared with {expected_random_positives:.1f} expected under uniform random selection "
        f"({lift_at_k:.2f}x lift)."
    )

    return OutreachResult(
        total_customers=n,
        outreach_ratio=outreach_ratio,
        k_selected=k,
        observed_positives=observed_positives,
        precision_at_k=precision_at_k,
        baseline_prevalence=baseline_prevalence,
        expected_random_positives=expected_random_positives,
        lift_at_k=lift_at_k,
        positive_capture_rate=capture_rate,
        interpretation=interpretation,
    )


def compute_cumulative_gains(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    steps: int = 20,
) -> pd.DataFrame:
    """Compute cumulative gains and lift across a grid of outreach budgets.

    Useful for plotting and comparing curves.
    """
    ratios = np.linspace(0.05, 1.0, steps)
    rows = []
    for r in ratios:
        res = evaluate_outreach(y_true, y_scores, outreach_ratio=float(r))
        rows.append(res.to_dict())
    return pd.DataFrame(rows)

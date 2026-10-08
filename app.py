"""CampaignIQ — Banking Customer Decision Intelligence.

Interactive decision-support prototype demonstrating customer outreach prioritization
using historical Portuguese bank marketing data.
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.decision_engine import evaluate_outreach

# Page configuration
st.set_page_config(
    page_title="CampaignIQ — Banking Decision Intelligence",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Financial Analytics CSS Styling
st.markdown(
    """
    <style>
    /* Hide Streamlit header chrome */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden;
    }
    
    /* Warm, clean aesthetic */
    .stApp {
        background-color: #FAFAFA;
        color: #1E293B;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header typography */
    h1 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    
    h2, h3 {
        color: #1E3A8A;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* Section Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }

    .question-box {
        background: #F1F5F9;
        border-left: 4px solid #1E3A8A;
        padding: 12px 18px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 12px;
        font-size: 1.05rem;
        color: #0F172A;
        font-weight: 500;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748B;
    }

    /* Pipeline badge flow */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin: 12px 0 18px 0;
    }
    .pipeline-step {
        background: #EFF6FF;
        color: #1E40AF;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #BFDBFE;
    }
    .pipeline-arrow {
        color: #94A3B8;
        font-weight: bold;
    }

    /* Footer styling */
    .footer-text {
        color: #64748B;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts():
    """Load pre-computed model artifacts from disk."""
    artifacts_dir = "artifacts"
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    preds_path = os.path.join(artifacts_dir, "test_predictions.csv")
    fi_path = os.path.join(artifacts_dir, "feature_importance.csv")

    if not os.path.exists(metrics_path) or not os.path.exists(preds_path):
        st.error("Artifacts missing. Please run `python scripts/reproduce.py` first.")
        st.stop()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    preds_df = pd.read_csv(preds_path)
    fi_df = pd.read_csv(fi_path) if os.path.exists(fi_path) else pd.DataFrame()

    return metrics, preds_df, fi_df


metrics, preds_df, fi_df = load_artifacts()
y_true = preds_df["y_true"].to_numpy()
y_scores = preds_df["y_score"].to_numpy()

# ==============================================================================
# SECTION 1 — Project Introduction
# ==============================================================================
st.markdown("<h1>CampaignIQ</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size: 1.15rem; color: #475569; font-weight: 500; margin-bottom: 4px;'>"
    "Banking Customer Decision Intelligence"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size: 0.95rem; color: #334155; margin-bottom: 12px;'>"
    "An ML-powered prototype that analyzes historical banking campaigns and demonstrates smarter customer outreach prioritization."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="question-box">
        💡 <strong>Core Business Question:</strong> What if a bank could prioritize customers most likely to respond instead of contacting everyone equally?
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr style='margin: 16px 0 20px 0; border: none; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2 — Evidence at a Glance
# ==============================================================================
st.markdown("### Evidence at a Glance")

total_dataset_size = 41188
test_samples = len(y_true)
selected_model_name = metrics["model_selection"]["selected_model"]
test_pr_auc = metrics["ranking_metrics"]["average_precision_pr_auc"]
default_p20 = metrics["primary_outreach_result"]
lift_at_default = default_p20["lift_at_k"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">{total_dataset_size:,}</div>
            <div class="metric-sub">{test_samples:,} in held-out test</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Selected Model</div>
            <div class="metric-value" style="font-size: 1.05rem; line-height: 1.25; padding-top: 6px; padding-bottom: 4px; font-weight: 700;">Logistic<br>Regression</div>
            <div class="metric-sub">Calibrated baseline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Test PR-AUC</div>
            <div class="metric-value">{test_pr_auc:.3f}</div>
            <div class="metric-sub">Average Precision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Lift @ 20% Budget</div>
            <div class="metric-value">{lift_at_default:.2f}x</div>
            <div class="metric-sub">vs. random expectation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Data source: UCI Bank Marketing dataset (41,188 observations, Portuguese retail bank term deposit campaigns). "
    "Chronological split (70% train / 15% validation / 15% test). All evaluation numbers reflect genuine held-out test performance."
)

st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)

# ==============================================================================
# SECTION 3 — Interactive Decision Simulator
# ==============================================================================
st.markdown("### What if you contact only the most promising customers?")
st.markdown(
    "<p style='color: #475569; font-size: 0.95rem;'>"
    "Simulate allocating marketing outreach capacity. Choose what percentage of the customer base your team has the budget to contact."
    "</p>",
    unsafe_allow_html=True,
)

# Single primary interactive control
outreach_pct = st.slider(
    label="Percentage of customers to prioritize:",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    format="%d%%",
)

outreach_ratio = outreach_pct / 100.0
sim_result = evaluate_outreach(y_true, y_scores, outreach_ratio=outreach_ratio)

# Key simulation numbers
s_col1, s_col2, s_col3, s_col4 = st.columns(4)

with s_col1:
    st.metric("Customers Selected (K)", f"{sim_result.k_selected:,}", f"{outreach_pct}% of {sim_result.total_customers:,}")

with s_col2:
    st.metric("ML Observed Positives", f"{sim_result.observed_positives:,}", f"{sim_result.precision_at_k:.1%} precision")

with s_col3:
    st.metric("Random Expected Positives", f"{sim_result.expected_random_positives:.1f}", f"{sim_result.baseline_prevalence:.1%} baseline")

with s_col4:
    st.metric("Prioritization Lift", f"{sim_result.lift_at_k:.2f}x", f"+{sim_result.observed_positives - sim_result.expected_random_positives:.0f} net positive gap")

# Single clean horizontal bar chart comparing ML vs Random
fig = go.Figure()

categories = ["Random-Selection Expectation", "ML-Prioritized Selection"]
positive_counts = [sim_result.expected_random_positives, sim_result.observed_positives]
colors = ["#94A3B8", "#1E3A8A"]

fig.add_trace(
    go.Bar(
        y=categories,
        x=positive_counts,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[
            f"{sim_result.expected_random_positives:.1f} positives ({sim_result.baseline_prevalence:.1%} rate)",
            f"{sim_result.observed_positives:,} positives ({sim_result.precision_at_k:.1%} rate)",
        ],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#FFFFFF", size=13, family="sans-serif"),
        hoverinfo="x+y",
    )
)

fig.update_layout(
    margin=dict(l=10, r=20, t=10, b=40),
    height=190,
    xaxis=dict(
        title="Positive Outcomes Identified (under same contact budget K)",
        showgrid=True,
        gridcolor="#E2E8F0",
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=False,
        tickfont=dict(size=12, color="#0F172A", family="sans-serif"),
    ),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
)

st.plotly_chart(fig, use_container_width=True)

# Plain English result interpretation
st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 12px 16px; margin: 8px 0 16px 0; font-size: 0.95rem; color: #1E293B;">
        <strong>Simulation Interpretation:</strong> At a <strong>{outreach_pct}%</strong> outreach budget (selecting <strong>{sim_result.k_selected:,}</strong> customers out of {sim_result.total_customers:,}), 
        the model's highest-ranked customers contained <strong>{sim_result.observed_positives:,}</strong> historical positive outcomes, 
        compared with <strong>{sim_result.expected_random_positives:.1f}</strong> expected under uniform random selection (<strong>{sim_result.lift_at_k:.2f}x lift</strong>).
        <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">
            *Note: Evaluation is computed on historical contacted records. This demonstrates ranking efficiency over random selection, but does not claim causal campaign uplift or guarantee future conversion rates without randomized controlled trials.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)

# ==============================================================================
# SECTION 4 — How the Intelligence Works
# ==============================================================================
st.markdown("### How the Intelligence Works")

# Pipeline representation
st.markdown(
    """
    <div class="pipeline-container">
        <span class="pipeline-step">Historical Data</span>
        <span class="pipeline-arrow">➔</span>
        <span class="pipeline-step">Leakage-Safe Cleaning</span>
        <span class="pipeline-arrow">➔</span>
        <span class="pipeline-step">Model Training & Validation</span>
        <span class="pipeline-arrow">➔</span>
        <span class="pipeline-step">Temporal Evaluation</span>
        <span class="pipeline-arrow">➔</span>
        <span class="pipeline-step">Customer Ranking Engine</span>
    </div>
    """,
    unsafe_allow_html=True,
)

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("##### 1. Model Selection & Performance")
    st.markdown(
        f"""
        - **Selected Model:** {selected_model_name}
        - **Why Selected:** Evaluated on validation data (PR-AUC: {metrics['model_selection']['logistic_regression']['val_average_precision']:.3f} vs {metrics['model_selection']['gradient_boosting']['val_average_precision']:.3f} for Gradient Boosting). It exhibited superior ranking stability and calibration across macroeconomic regime shifts.
        - **Test PR-AUC:** {test_pr_auc:.3f}
        - **Test ROC-AUC:** {metrics['ranking_metrics']['roc_auc']:.3f}
        - **Test Balanced F1:** {metrics['classification_metrics']['f1_score']:.3f} (threshold: {metrics['classification_metrics']['decision_threshold']:.2f})
        """
    )

with info_col2:
    st.markdown("##### 2. Key Influential Drivers (Top 5)")
    if not fi_df.empty:
        top_5 = fi_df.head(5)
        for i, row in top_5.iterrows():
            st.markdown(f"- **{row['feature']}**: {row['importance_mean']:.4f} validation permutation impact")
    st.caption("Feature importance indicates how strongly features affect model performance. It does not prove causal relationships.")

st.markdown("##### 3. Important Methodological Limitations")
st.markdown(
    """
    1. **Temporal & Macroeconomic Distribution Shift:** In the chronological split, positive response prevalence rose from 5.6% (Train, 2008) to 38.5% (Test, late 2009–2010), reflecting the Eurozone debt crisis and shifting interest rate dynamics.
    2. **Observational Nature:** All customers in the historical dataset were contacted. Ranking shows prioritization efficiency on historically observed responses, not causal uplift or incremental response from unseen marketing treatments.
    """
)

st.markdown(
    """
    <div style="font-size: 0.85rem; color: #475569; margin-top: 10px;">
        📖 For complete methodology, feature definitions, and reproducible code, see the 
        <a href="https://github.com" target="_blank" style="color: #1E40AF; text-decoration: underline;">GitHub Repository & Documentation</a>.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    """
    <div class="footer-text">
        <strong>CampaignIQ</strong> — Independent educational data science project<br>
        Source: UCI Machine Learning Repository (Bank Marketing Dataset)<br>
        Research prototype. Not a production banking decision system.
    </div>
    """,
    unsafe_allow_html=True,
)

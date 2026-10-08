# Technical Methodology — CampaignIQ

**Project:** CampaignIQ — Banking Customer Decision Intelligence  
**Focus:** Pre-contact predictive ranking and decision-support modeling for term deposit outreach.  
**Dataset Source:** UCI Machine Learning Repository — Bank Marketing (`bank-additional-full.csv`)

---

## 1. Problem Formulation & Prediction Point

Financial institutions dedicate finite call center and relationship manager hours to outreach campaigns. Contacting every customer uniformly is inefficient, especially when positive response prevalence is low.

**Objective:** Predict the likelihood that an existing retail bank customer will subscribe to a term deposit (`y = 1`) prior to placing a marketing phone call, enabling optimized customer outreach ranking.

**Critical Prediction Boundary:**
- **Prediction Point:** Prior to placing the telephone call.
- **Allowed Features:** Only features strictly observable before contact is initiated (demographics, prior relationship history, and prevailing macroeconomic indicators).
- **Prohibited Features:** Any interaction details collected during or after the call.

---

## 2. Target Leakage Prevention

Data leakage is the inclusion of information in training data that would not be realistically available at the time of inference. Leakage artificially inflates evaluation metrics and leads to failure upon live deployment.

### 2.1 Call Duration (`duration`)
In the raw UCI dataset, `duration` represents the duration of the phone call in seconds. As the original dataset authors (Moro et al., 2014) note:
> *"The call duration is not known before a call is performed. Also, after the end of the call, `y` is known. Thus, this input should only be included for benchmark purposes and must be discarded if the intention is to have a realistic predictive model."*

Including `duration` allows a model to simply learn that calls lasting over 5 minutes result in subscriptions, creating severe target leakage. **`duration` is strictly removed prior to all modeling.**

### 2.2 Current Campaign Contacts (`campaign`)
The feature `campaign` represents the total number of contacts performed during the current campaign for this client, including the current call. Because its value depends on the sequence and progress of contacts within the current active campaign, it creates ambiguity regarding pre-contact availability. To guarantee strict pre-contact feature purity, **`campaign` is excluded from the feature set.**

### 2.3 Feature Separation Table

| Feature Category | Features Included | Availability at Prediction Point | Rationale |
| :--- | :--- | :--- | :--- |
| **Customer Profile** | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` | Available in core banking system | Known pre-contact customer demographics and credit attributes. |
| **Outreach Channel & Timing** | `contact`, `month`, `day_of_week` | Available at planning stage | Campaign planning metadata (scheduled channel and contact timing). |
| **Historical Interaction** | `pdays`, `previous`, `poutcome` | Available in CRM database | Results from preceding campaigns (days since last contact, past outcome). |
| **Macroeconomic Indicators** | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Public economic indicators | Published indicators (quarterly/monthly/daily) reflecting market conditions. |
| **Excluded (Leakage)** | `duration`, `campaign` | **UNAVAILABLE PRE-CONTACT** | Call duration occurs during the call; current campaign contact counts reflect in-flight operations. |

---

## 3. Chronological Data Splitting

The UCI Bank Marketing dataset is ordered chronologically by contact date (May 2008 to November 2010).

Rather than applying random k-fold cross-validation—which creates look-ahead bias by mixing future records into training sets—we employ a strict **chronological train/validation/test split**:

1. **Training Set (First 70%):** 28,831 records (May 2008 to mid-2009)
2. **Validation Set (Next 15%):** 6,178 records (mid-2009 to early 2010)
3. **Held-out Test Set (Final 15%):** 6,179 records (mid-2010 to November 2010)

### Empirical Findings: Severe Macroeconomic Distribution Shift
Chronological splitting exposes an important real-world financial analytics phenomenon:

- **Train Positive Prevalence:** **5.57%** (1,607 subscriptions out of 28,831)
- **Validation Positive Prevalence:** **10.65%** (658 subscriptions out of 6,178)
- **Test Positive Prevalence:** **38.45%** (2,376 subscriptions out of 6,179)

**Root Cause:** Between 2008 and 2010, the European sovereign debt crisis unfolded. Euribor 3-month interest rates fell dramatically from ~4.85% to under 1.0%, employment rates dropped, and Portuguese retail customers increasingly sought the capital preservation and guaranteed yield of fixed-term bank deposits. Furthermore, the bank refined its targeting strategies in later campaign phases.

*A random split would have masked this critical macroeconomic shift by blending the periods to an average 11.2% rate, producing misleadingly optimistic cross-validation claims.*

---

## 4. Preprocessing & Pipeline Architecture

All transformations are encapsulated in a `scikit-learn` `Pipeline` and `ColumnTransformer`. Preprocessing parameters (medians, means, scalers, one-hot category dictionaries) are **fitted strictly on the training set** to prevent data snooping.

- **Numerical Features (8):**
  - Missing values imputed using `SimpleImputer(strategy='median')`.
  - Scaled using `StandardScaler()`.
- **Categorical Features (10):**
  - Explicit `"unknown"` categories handled natively.
  - Imputed using `SimpleImputer(strategy='constant', fill_value='unknown')`.
  - Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to safely ingest unseen levels during production inference without runtime crashes.

---

## 5. Model Development & Validation Comparison

We trained two models using balanced class weighting to address positive class sparsity:

1. **Model A — Regularized Logistic Regression:**
   - Solver: `lbfgs`, `max_iter=1000`, `class_weight='balanced'`.
   - Highly interpretable linear log-odds scoring.
2. **Model B — HistGradientBoostingClassifier:**
   - Native histogram-based gradient boosting (`learning_rate=0.08`, `min_samples_leaf=25`, `class_weight='balanced'`).
   - Models non-linear interaction effects.

### Validation Set Comparison

| Metric | Logistic Regression (Model A) | HistGradientBoosting (Model B) | Selection |
| :--- | :---: | :---: | :---: |
| **Validation PR-AUC (Average Precision)** | **0.1870** | 0.1442 | **Model A (+29.7% higher)** |
| **Validation ROC-AUC** | **0.6566** | 0.5777 | **Model A** |
| **Validation Brier Score (Calibration)** | **0.3599** | 0.4011 | **Model A (Lower error)** |
| **Optimal Validation F1** | **0.2807** | 0.2225 | **Model A** |
| **Optimal Validation Threshold** | **0.6957** | 0.7461 | Chosen for binary classification |

**Selection Rationale:**  
Under the severe macroeconomic distribution shift between training (2008) and validation (2009), the tree-based gradient booster overfit to the early macroeconomic environment, resulting in degraded ranking on future data. Logistic Regression exhibited superior generalization, higher Average Precision, and lower Brier score error. **Logistic Regression was selected as the champion model.**

---

## 6. Held-Out Test Evaluation

The champion model was evaluated once on the untouched held-out test set (6,179 records):

| Evaluation Dimension | Metric | Score | Analytical Interpretation |
| :--- | :--- | :---: | :--- |
| **Ranking Performance** | **Average Precision (PR-AUC)** | **0.4371** | Primary ranking metric; robust to positive class prevalence. |
| | **ROC-AUC** | **0.5592** | Global pairwise ranking discrimination across full probability spectrum. |
| **Threshold Classification** | **Accuracy (at threshold 0.696)** | **57.24%** | Naive majority-class accuracy is 61.55% (predicting all 0s). Demonstrates why accuracy is an uninformative metric for banking campaigns. |
| | **Precision** | **45.16%** | Probability that a model-flagged customer actually subscribes. |
| | **Recall** | **52.23%** | Proportion of total subscribers captured. |
| | **F1 Score** | **0.4844** | Harmonic mean of precision and recall. |
| **Calibration** | **Brier Score** | **0.3352** | Mean squared difference between predicted probabilities and observed binary outcomes. |

---

## 7. Decision Intelligence Engine & Formulations

The Decision Intelligence Engine (`src/decision_engine.py`) translates predicted probabilities into outreach prioritization.

Given $N$ customers in the held-out test cohort and user-selected contact budget ratio $r \in (0, 1]$:

$$K = \text{round}(N \times r)$$

1. **Ranking:** Sort all observations descending by predicted probability:
   $$\hat{s}_{(1)} \ge \hat{s}_{(2)} \ge \dots \ge \hat{s}_{(N)}$$
2. **Top-K Selection:** Select top $K$ ranked customers:
   $$\mathcal{S}_K = \{(x_{(i)}, y_{(i)}) \mid 1 \le i \le K\}$$
3. **Observed Positives:**
   $$\text{Positives}_K = \sum_{i=1}^K y_{(i)}$$
4. **Precision-at-K:**
   $$\text{Precision@K} = \frac{\text{Positives}_K}{K}$$
5. **Baseline Prevalence:**
   $$p_{\text{base}} = \frac{1}{N} \sum_{i=1}^N y_i$$
6. **Expected Random Positives:**
   $$\mathbb{E}[\text{Positives}_{\text{random}}] = K \times p_{\text{base}}$$
7. **Lift-at-K:**
   $$\text{Lift@K} = \frac{\text{Precision@K}}{p_{\text{base}}} = \frac{\text{Positives}_K}{\mathbb{E}[\text{Positives}_{\text{random}}]}$$

### Decision Performance Across Outreach Budgets

| Budget ($r$) | Customers Contacted ($K$) | Observed Positives | Expected Random Positives | Precision@K | Lift@K | Positives Captured |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5%** | 309 | 159 | 118.8 | **51.46%** | **1.34x** | 6.69% |
| **10%** | 618 | 296 | 237.6 | **47.90%** | **1.25x** | 12.46% |
| **15%** | 927 | 425 | 356.5 | **45.85%** | **1.19x** | 17.89% |
| **20%** | 1,236 | 527 | 475.3 | **42.64%** | **1.11x** | 22.18% |
| **30%** | 1,854 | 809 | 712.9 | **43.64%** | **1.13x** | 34.05% |
| **50%** | 3,090 | 1,365 | 1,188.2 | **44.17%** | **1.15x** | 57.45% |

---

## 8. Global Feature Explainability & Failure Analysis

### 8.1 Permutation Importance (Validation Set)
Computed using 10 repeats with Average Precision scoring on the validation set:

1. **`month` (0.0313 ± 0.005):** Timing and seasonality are primary drivers; bank campaigns in late autumn and spring had markedly distinct response rates.
2. **`pdays` (0.0229 ± 0.005):** Number of days since client was last contacted; recent contacts strongly predict receptivity.
3. **`contact` (0.0051 ± 0.002):** Cellular outreach outperformed fixed landline telephone outreach.
4. **`marital` (0.0041 ± 0.002):** Marital status correlation with financial stage of life.
5. **`poutcome` (0.0024 ± 0.002):** Prior campaign success (`poutcome = 'success'`) provides a strong affirmative signal.

> *Disclaimer: Permutation feature importance measures sensitivity of the model's loss function to feature shuffle. It indicates empirical correlation within this dataset, not causal mechanisms.*

### 8.2 Failure Pattern Analysis
- **High-Confidence False Positives:** Young clients (ages 22–30, students/technicians) with prior campaign successes during low interest rate periods who nonetheless declined the deposit. Model over-indexed on prior success despite client liquidity constraints.
- **Low-Confidence False Negatives:** Older clients (ages 55–85, retired/housemaids) who had previously declined an offer (`poutcome = 'failure'`), yet subscribed during the late 2010 crisis when capital security was prioritized.

---

## 9. Key Analytical Limitations

1. **Historical Portuguese Dataset:** Records span 2008–2010 from a Portuguese retail bank. Findings cannot be directly mapped to contemporary institutions or emerging retail banking environments (such as Indian small finance banks) without local data calibration.
2. **Observational vs. Causal Uplift:** Every record in this dataset received a marketing call. The model estimates $P(Y=1 \mid X, \text{Contacted}=1)$. It cannot estimate true treatment effect (incremental uplift) without randomized uncontacted control groups.
3. **No Financial Cost Assumptions:** Without real cost-per-call, agent wage, and deposit net interest margin (NIM) data, we intentionally omit speculative dollar-value ROI claims.

# Interview Guide — CampaignIQ

**Target Role:** Data Science & Decision Management Internship  
**Project:** CampaignIQ — Banking Customer Decision Intelligence  

---

## 1. 60–90 Second Interview Elevator Pitch

> *"In retail banking, relationship managers and call center teams have finite hours to reach out to customers about savings and term deposits. Uniformly calling every customer or randomly selecting lists wastes time and annoys uninterested clients.*
>
> *I designed **CampaignIQ** to solve this prioritization problem. It’s an educational decision intelligence prototype built on 41,188 historical bank marketing records.*
>
> *What sets this project apart is rigorous analytical discipline:*
> *First, I enforced strict leakage prevention by dropping call duration—which is only known after a call takes place—and contact-in-flight attributes.*
> *Second, I used a chronological train/validation/test split rather than random cross-validation. This revealed a real macroeconomic regime shift during the 2008–2010 financial crisis, where positive response prevalence rose from 5.6% to 38.5% as Euribor interest rates plummeted.*
> *Third, by comparing Logistic Regression and Gradient Boosting on validation PR-AUC and calibration, Logistic Regression emerged as the superior generalizer.*
>
> *Finally, I translated the model into a Decision Engine and interactive simulator: at a 20% outreach budget, prioritising the top-ranked customers identified 527 historical deposit subscribers compared to 475.3 expected under uniform random targeting—delivering a 1.11x lift without inventing speculative conversion claims.*
>
> *This taught me that in financial analytics, sound methodology, leakage prevention, and decision framing matter far more than raw algorithmic complexity."*

---

## 2. Core Technical & Business Interview Questions

### Q1: What business problem did this project solve?
**Answer:** It addresses marketing outreach inefficiency in retail banking. Rather than contacting customers uniformly or at random, the project uses pre-contact customer and economic attributes to rank customers by propensity, maximizing the concentration of positive responses within a capped outreach budget.

### Q2: Where did the dataset come from?
**Answer:** The official UCI Machine Learning Repository Bank Marketing dataset (`bank-additional-full.csv`), comprising 41,188 observations from a Portuguese retail bank from May 2008 to November 2010.

### Q3: Why did we use machine learning instead of basic business heuristics?
**Answer:** Traditional business heuristics (e.g., "only call clients over age 50") evaluate features in isolation and miss multivariate interactions between macroeconomic climate (Euribor rates, price indices), prior campaign recency (`pdays`), and demographic profiles. ML learns a calibrated joint scoring function.

### Q4: Why did we compare two algorithms?
**Answer:** Good data science practice requires establishing a clear, interpretable baseline before adopting complex models. We compared a transparent linear model (Logistic Regression) against a non-linear tree ensemble (Gradient Boosting) on held-out validation data to determine whether non-linear complexity justified itself.

### Q5: What is Logistic Regression in this context?
**Answer:** It models the log-odds of a binary outcome (subscribing to a term deposit) as a linear combination of input features, passing the result through a sigmoid function to output a probability between 0 and 1. It provides stable coefficients, resistance to severe overfitting, and calibrated ranking.

### Q6: What is Gradient Boosting in this context?
**Answer:** An ensemble technique that builds decision trees sequentially, where each successive tree is trained to predict the residual errors (gradient of the loss function) of the preceding trees. While powerful, it is prone to fitting historical macro regimes that may shift over time.

### Q7: What is data leakage?
**Answer:** Target leakage occurs when features included during training contain information that would not realistically be available at the exact moment predictions must be made in production. It results in artificially inflated test scores that fail catastrophically in deployment.

### Q8: Why did we strictly exclude call duration?
**Answer:** Call duration (`duration`) is only known *after* a phone call finishes. A customer cannot have a 10-minute call unless they stay on the line to discuss the product. Using duration before dialing the phone is impossible and creates massive target leakage.

### Q9: Why did we use a chronological split instead of random k-fold cross-validation?
**Answer:** Bank campaign data is inherently sequential. Random splitting mixes future and past records, causing look-ahead bias and masking macroeconomic trends. A chronological split (first 70% train, next 15% val, last 15% test) mimics real banking deployment and exposed a major macroeconomic shift between 2008 and 2010.

### Q10: What does Average Precision (PR-AUC) measure, and why prefer it over accuracy?
**Answer:** In imbalanced datasets, classification accuracy is misleading—predicting "no" for all customers can yield high accuracy while identifying zero actual subscribers. Average Precision computes the area under the Precision-Recall curve across all decision thresholds, directly measuring how effectively the model ranks positive instances without being distorted by the majority class.

### Q11: What is Precision-at-K?
**Answer:** It is the proportion of actual positive outcomes among the top $K$ customers selected by the model: $\text{Precision@K} = \frac{\text{Observed Positives in Top } K}{K}$. It directly reflects the response rate a call center team experiences when contacting their top leads.

### Q12: How does the Decision Intelligence Engine work?
**Answer:** Given a user-specified outreach budget (e.g., 20% of customers), the engine ranks all test customers descending by predicted score, selects the top $K$ individuals, counts how many actually subscribed, and compares that count against the mathematical expectation of selecting $K$ customers uniformly at random.

### Q13: What does "lift" mean?
**Answer:** Lift is the factor by which the model-guided response rate exceeds the baseline response rate: $\text{Lift} = \frac{\text{Precision@K}}{\text{Baseline Prevalence}}$. A lift of 1.25x means the prioritized list is 25% more concentrated with positive responses than random outreach.

### Q14: What are the project’s primary limitations?
**Answer:**
1. **Historical Portuguese Data:** Collected during 2008–2010 during the European debt crisis; not directly portable to contemporary banking or Indian retail markets without local retraining.
2. **Observational, Not Causal:** All records were historically contacted. The model estimates conditional response given contact, but cannot prove *causal campaign uplift* (i.e. who only bought *because* of the call) without an uncontacted randomized control trial.
3. **Absence of Cost Data:** Without true call center wage costs and deposit interest margins, we intentionally avoided speculative financial ROI projections.

### Q15: What would be necessary to deploy this in a real bank?
**Answer:**
1. Real-time API integration with the core banking system / CRM.
2. An A/B test (randomized champion-challenger campaign) with an uncontacted holdout control group to measure true causal incremental lift.
3. Automated data drift and concept drift monitoring (tracking Euribor/repo rates, unemployment, and population response rates).
4. Fair lending and ethical bias audits to ensure prioritization does not unfairly exclude protected demographic segments.

---

## 3. Decisions for Personal Student Ownership & Review

As the project owner directing this work, you should personally review and approve:

1. **Prediction Point Boundary:** Verify that only features available before dialing (`age`, `job`, `euribor3m`, `pdays`, etc.) were retained and `duration` and `campaign` were dropped.
2. **Model Selection Choice:** Understand why Logistic Regression was selected over Gradient Boosting (validation PR-AUC of 0.1870 vs 0.1442 due to tree overfitting across macro regime shifts).
3. **Outreach Budget Default:** Confirm the default outreach budget slider is set to 20% (selecting 1,236 customers, achieving 42.6% precision vs 38.5% random baseline, yielding 1.11x lift).
4. **Honest Framing of Results:** Ensure no claims are made about "guaranteeing 20% higher future deposits" or "reducing bank operational costs by $50k," adhering strictly to evidence-based historical evaluation.
5. **Contribution Statement:** Understand the project contribution statement:
   *"Independent project focused on financial problem framing, analytical design, model evaluation, decision modelling, and product testing. Implementation developed with AI-assisted coding tools and validated through reproducible experiments."*

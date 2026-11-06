# 🏦 Credit Risk Modelling — PD Scorecard
> **A probability-of-default scorecard for lenders whose borrowers have thin or no credit-bureau history. Predicts default probability, converts it into a 300–900 credit score, and returns an interpretable risk rating.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-purple)](https://optuna.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Live Demo

👉 [Credit Risk Modelling Streamlit App](https://creditriskmodeling-dbvgyaavxuqjfbjtztxep5.streamlit.app/)

Enter a borrower profile and the app returns:

- **Default probability** — the model's estimated likelihood of default
- **Credit score** — that probability rescaled onto a 300–900 range
- **Rating band** — Poor / Average / Good / Excellent

---

## 📌 Problem Statement

Small finance companies and NBFCs lend to a borrower base that often lacks formal credit
history, which makes bureau-score-based underwriting insufficient. This project builds a
**binary classification scorecard** that estimates the probability a given applicant will
default, using application and behavioural data rather than a bureau score alone.

The output is designed to support an origination decision — approve, review, or decline —
and to be explainable to a credit officer.

---

## 🎯 Scope

**What this project does**

- Estimates **Probability of Default (PD)** from 13 application and behavioural features
- Converts PD into a 300–900 credit score with four rating bands
- Serves predictions through a Streamlit application

**What it deliberately does not do**

- It does **not** model Loss Given Default (LGD) or Exposure at Default (EAD)
- It does **not** compute Expected Credit Loss (ECL) or IFRS 9 provisions
- It is a single-model PD scorecard, not a full credit-loss engine

Those would be the natural next stage; see [Roadmap](#-roadmap).

---

## 🗂️ Repository Structure

```
credit_risk_modeling/
│
├── credit_risk_model.ipynb              # Main notebook: EDA → features → training → evaluation
├── optuna_tutorial.ipynb                # Hyperparameter optimisation walkthrough
│
├── app/
│   ├── main.py                          # Streamlit UI
│   ├── prediction_helper.py             # Preprocessing + scoring logic
│   └── artifacts/model_data.joblib      # Copy of the canonical artifact
│
├── artifacts/
│   └── model_data.joblib                # CANONICAL — written by the notebook's final cell
│
├── dataset/
│   ├── customers.csv                    # Demographics and application data
│   ├── loans.csv                        # Loan-level attributes and default flag
│   └── bureau_data.csv                  # Credit bureau attributes
│
├── requirements.txt
└── Credit Risk Model Small Finance Company.pptx
```

> **Note on artifacts:** `artifacts/model_data.joblib` at the repo root is canonical —
> it is what the notebook writes. `app/artifacts/` holds an identical copy.
> `prediction_helper.py` resolves the path relative to its own file location, so the
> same model loads regardless of the working directory.

---

## 🧠 Modelling Approach

A single-stage binary classifier. The target is a default flag; the output is a calibrated
probability that is then rescaled into a score.

```
Raw application / loan / bureau data
            │
            ▼
   Train/test split  (performed BEFORE cleaning, to avoid leakage)
            │
            ▼
   Cleaning  →  missing values, duplicates, outlier treatment,
                data-error correction in loan_purpose
            │
            ▼
   Feature engineering  →  loan-to-income, delinquency ratio,
                           avg DPD per delinquency
            │
            ▼
   Feature selection  →  VIF (multicollinearity)
                         WOE / Information Value (categorical)
            │
            ▼
   Encoding + MinMax scaling
            │
            ▼
   Model search  →  Logistic Regression vs XGBoost
                    RandomizedSearchCV, then Optuna
                    Imbalance: RandomUnderSampler, SMOTETomek,
                               class_weight, scale_pos_weight
            │
            ▼
   Evaluation  →  ROC/AUC, Gini, KS, decile rank-ordering
            │
            ▼
   FINAL MODEL: Logistic Regression  →  artifacts/model_data.joblib
```

### Why logistic regression as the final model

XGBoost scored marginally higher on the tuned F1 objective (0.975 vs 0.946 in the Optuna
trials), but logistic regression was selected because:

- Coefficients are directly interpretable, which matters for adverse-action reasoning
- It is the established standard for regulated credit scorecards
- The accuracy difference did not justify the loss of explainability

This mirrors standard industry practice: use the boosted model as a challenger and for
feature discovery, ship the interpretable one.

---

## 📊 Features Used (13)

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age |
| `loan_tenure_months` | Numeric | Requested loan term |
| `number_of_open_accounts` | Numeric | Active credit accounts |
| `credit_utilization_ratio` | Numeric | Outstanding debt ÷ credit limit |
| `loan_to_income` | Engineered | Loan amount ÷ annual income |
| `delinquency_ratio` | Engineered | Past dues ÷ total payments |
| `avg_dpd_per_delinquency` | Engineered | Average days past due per delinquent event |
| `residence_type_Owned` | One-hot | Reference category: Mortgage |
| `residence_type_Rented` | One-hot | |
| `loan_purpose_Education` | One-hot | Reference category: Auto |
| `loan_purpose_Home` | One-hot | |
| `loan_purpose_Personal` | One-hot | |
| `loan_type_Unsecured` | One-hot | Reference category: Secured |

---

## 📈 Model Performance

Evaluated on a held-out test set of **12,497 records** (train: 37,500; default rate ≈ 8.6%).

| Metric | Value |
|---|---|
| ROC-AUC | **0.984** |
| Gini coefficient | **0.967** |
| Accuracy | 0.93 |
| Recall (default class) | **0.94** |
| Precision (default class) | 0.56 |
| F1 (default class) | 0.70 |

KS statistic and rank-ordering are computed per decile in the notebook's evaluation section.

> ### ⚠️ On the AUC
>
> An AUC of 0.98 is **unusually high for credit default prediction**, where 0.75–0.85 is
> typical. This is worth investigating before treating the model as production-ready.
> The most likely explanation is **target leakage through the behavioural features**:
> `delinquency_ratio` and `avg_dpd_per_delinquency` are derived from repayment history,
> and if that history spans or follows the default event, they partly encode the target.
>
> A proper fix is to compute those features **strictly from data available at the point of
> application**, then re-measure. See [Roadmap](#-roadmap).

### Class-imbalance trade-off

The model is tuned for **recall on the default class** (0.94) at the cost of precision
(0.56) — roughly 44% of flagged applications are false positives. For origination this is
usually the right direction, since a missed default costs the loan value while a false
positive costs a review. The operating point should be set from the business cost ratio
rather than from F1.

---

## 🎚️ Score Mapping

```python
credit_score = 300 + (1 - default_probability) * 600
```

| Score | Rating |
|---|---|
| 300–499 | Poor |
| 500–649 | Average |
| 650–749 | Good |
| 750–900 | Excellent |

> **Known limitation:** this mapping is linear in *probability*. Industry scorecards are
> linear in *log-odds* — `score = Offset + Factor × ln(odds)`, with `Factor = PDO / ln(2)`
> — which gives a constant odds interpretation per point across the whole range. The
> current scaling compresses discrimination at the extremes. The band cut-offs are also
> indicative rather than derived from swap-set analysis.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Modelling | scikit-learn (LogisticRegression), XGBoost (challenger) |
| Imbalance | imbalanced-learn — RandomUnderSampler, SMOTETomek |
| HPO | Optuna (TPE sampler), RandomizedSearchCV |
| App | Streamlit |
| Serialisation | joblib |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Sajalagarwal-ca/credit_risk_modeling.git
cd credit_risk_modeling
pip install -r requirements.txt
```

**Run the notebook**

```bash
jupyter notebook credit_risk_model.ipynb
```

**Run the app** — from the repository root:

```bash
streamlit run app/main.py
```

---

## 📋 Notebook Walkthrough

1. **Load and split** — three source tables joined; train/test split performed *before*
   cleaning so no test information influences preprocessing
2. **Cleaning** — missing values, duplicates, outlier removal on `processing_fee`,
   correction of data-entry errors in `loan_purpose`
3. **EDA** — box plots, KDE plots, categorical distribution analysis
4. **Feature engineering** — loan-to-income, delinquency ratio, average DPD per delinquency
5. **Feature selection** — VIF for multicollinearity; WOE and Information Value for
   categorical predictive power
6. **Model search** — four attempts across Logistic Regression and XGBoost, with
   RandomizedSearchCV then Optuna, testing RandomUnderSampler and SMOTETomek
7. **Evaluation** — ROC/AUC, Gini, decile rank-ordering and KS
8. **Finalise** — coefficient-based feature importance, artifact serialisation

---

## 🔮 Roadmap

**Correctness first**

- [ ] **Investigate the 0.98 AUC for target leakage** — recompute behavioural features
      using only data available at application time, then re-baseline
- [ ] Replace the dummy-column workaround in `prediction_helper.py` with a scikit-learn
      `Pipeline` / `ColumnTransformer` so preprocessing and model are serialised together
- [ ] Move the scorecard scaling into the notebook, derived from log-odds, rather than
      defining it in the app
- [ ] Add calibration assessment — a reliability curve and Brier score — since a
      probability that drives a decision needs to be right in *level*, not just in ranking

**Extensions**

- [ ] Add an **LGD model** and **EAD** assumption to enable `ECL = PD × LGD × EAD`
- [ ] Reject inference, to correct for training only on approved applicants
- [ ] SHAP explainability panel in the Streamlit app
- [ ] Drift monitoring — PSI on scores and features, actual-vs-expected by score band
- [ ] MLflow experiment tracking; Docker packaging; FastAPI inference endpoint

---

## 👤 Author

**Sajal Agarwal** — Toronto, Canada
GitHub: [@Sajalagarwal-ca](https://github.com/Sajalagarwal-ca)

---

## 📄 License

MIT — see [LICENSE](LICENSE).

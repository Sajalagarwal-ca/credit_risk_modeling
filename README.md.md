# 🏦 Credit Risk Modeling — Small Finance Company

> **End-to-end machine learning pipeline for predicting loan default probability and loss given default, tailored for small finance institutions with limited credit bureau coverage.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-purple)](https://optuna.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Problem Statement

Small finance companies and NBFCs face a unique challenge: their borrower base often lacks formal credit history, making traditional bureau-score-based underwriting insufficient. This project builds a **dual-output credit risk model** that simultaneously estimates:

- **Probability of Default (PD)** — will this borrower default?
- **Loss Given Default (LGD)** — if they default, how much will the lender lose?

The final output is an **Expected Credit Loss (ECL)** score used to approve/reject loan applications and set risk-adjusted interest rates.

---

## 🗂️ Repository Structure

```
credit_risk_modeling/
│
├── credit_risk_model_codebasics.ipynb   # Main modeling notebook (EDA → Feature Eng → Training → Evaluation)
├── optuna_tutorial.ipynb                # Hyperparameter optimization with Optuna
│
├── app/                                 # Streamlit web application
│   ├── app.py                           # Main app entry point
│   └── ...
│
├── artifacts/                           # Saved model objects & encoders
│   ├── model_pd.pkl                     # Trained PD classifier
│   ├── model_lgd.pkl                    # Trained LGD regressor
│   └── ...
│
├── dataset/                             # Raw and processed data
│   └── ...
│
├── requirements.txt                     # Python dependencies
└── Credit Risk Model Small Finance Company.pptx   # McKinsey-style project presentation
```

---

## 🧠 Model Architecture

The system follows a **two-stage pipeline**:

```
Raw Application Data
        │
        ▼
┌─────────────────────┐
│  Data Preprocessing  │  ← Missing value imputation, outlier treatment, encoding
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Feature Engineering │  ← Income ratios, credit utilization, loan-to-value, bureau scores
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌─────────┐
│  PD     │  │  LGD    │
│ Model   │  │ Model   │
│(XGBoost)│  │(XGBoost)│
└────┬────┘  └────┬────┘
     │             │
     └──────┬──────┘
            ▼
   ECL = PD × LGD × EAD
            │
            ▼
   ┌──────────────────┐
   │ Risk Score Output │  ← Approve / Reject / Manual Review
   └──────────────────┘
```

### Stage 1 — Probability of Default (PD)
- **Algorithm:** XGBoost Classifier
- **Target:** Binary (1 = default, 0 = no default)
- **Threshold tuning:** F1-optimized cutoff via ROC-AUC analysis
- **Handling imbalance:** `scale_pos_weight`, SMOTE-inspired weighting

### Stage 2 — Loss Given Default (LGD)
- **Algorithm:** XGBoost Regressor (or two-stage beta regression)
- **Target:** Recovery rate (0–1), transformed to LGD
- **Only applied** to predicted defaulters from Stage 1

### Hyperparameter Optimization
- **Framework:** Optuna with TPE sampler
- **Search space:** `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- **Pruning:** Median pruner for early stopping of unpromising trials

---

## 📊 Key Features Engineered

| Feature | Description |
|--------|-------------|
| `loan_to_income` | Loan amount / Annual income |
| `credit_utilization` | Outstanding debt / Credit limit |
| `delinquency_ratio` | Past dues / Total historical payments |
| `employment_stability` | Tenure at current employer (binned) |
| `bureau_score_band` | Discretized CIBIL/Experian score |
| `ltv_ratio` | Loan to collateral value (for secured loans) |
| `income_per_dependent` | Income normalized by family obligations |

---

## 📈 Model Performance

| Metric | PD Model | LGD Model |
|--------|----------|-----------|
| ROC-AUC | ~0.85+ | — |
| Gini Coefficient | ~0.70 | — |
| KS Statistic | ~0.45 | — |
| RMSE | — | ~0.12 |
| R² | — | ~0.68 |

> Metrics are indicative based on the training dataset. See notebook for full evaluation cells.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Modeling** | scikit-learn, XGBoost |
| **HPO** | Optuna |
| **App** | Streamlit |
| **Serialization** | Pickle / Joblib |
| **Notebook** | Jupyter Notebook |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Sajalagarwal-ca/credit_risk_modeling.git
cd credit_risk_modeling
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Open `credit_risk_model_codebasics.ipynb` in Jupyter:

```bash
jupyter notebook credit_risk_model_codebasics.ipynb
```

### 4. Launch the Streamlit App

```bash
cd app
streamlit run app.py
```

---

## 📋 Notebook Walkthrough

The main notebook covers:

1. **Exploratory Data Analysis (EDA)**
   - Default rate distribution, class imbalance analysis
   - Correlation heatmaps, feature distributions
   - Bivariate analysis: default rate vs. key predictors

2. **Data Preprocessing**
   - Missing value strategy (median/mode imputation)
   - Categorical encoding (target encoding, WoE)
   - Train/validation/test split (stratified)

3. **Feature Engineering**
   - Domain-driven ratio features
   - Interaction terms, polynomial features
   - Feature importance & selection via SHAP

4. **Model Training**
   - Baseline: Logistic Regression
   - Advanced: XGBoost with class weighting
   - Hyperparameter tuning via Optuna (`optuna_tutorial.ipynb`)

5. **Model Evaluation**
   - ROC-AUC, KS statistic, Gini coefficient
   - Confusion matrix, precision-recall tradeoff
   - SHAP waterfall/beeswarm plots for interpretability

6. **ECL Calculation**
   - Combining PD and LGD outputs
   - Portfolio-level expected loss simulation

---

## 🌐 Use Cases

- **Loan origination:** Real-time risk scoring at point of application
- **Portfolio monitoring:** Batch ECL computation for regulatory provisioning (IFRS 9)
- **Pricing engine:** Risk-adjusted rate setting for new borrowers
- **Collection prioritization:** Identify high-LGD accounts for early intervention

---

## 🔮 Future Enhancements

- [ ] Add SHAP-based explainability dashboard in Streamlit
- [ ] Incorporate time-series delinquency features (vintage analysis)
- [ ] MLflow experiment tracking
- [ ] Docker containerization for deployment
- [ ] REST API wrapper (FastAPI)
- [ ] Model drift monitoring (Evidently AI)

---

## 👤 Author

**Sajal Agarwal**
- GitHub: [@Sajalagarwal-ca](https://github.com/Sajalagarwal-ca)
- Location: Toronto, Canada

---

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ for the ML & fintech community. Star ⭐ the repo if you found it useful!*

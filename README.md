# 📡 Telco Customer Churn Prediction and Behavior Analytics

> **IBM SkillsBuild Data Analytics with AI Academic Internship** — BharatCares × AICTE  
> **Candidate:** Vedant &nbsp;|&nbsp; **Dataset:** IBM Telco Customer Churn &nbsp;|&nbsp; **Models:** Logistic Regression · Random Forest

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.2.2-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Academic%20Use-green)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Overview](#-dataset-overview)
3. [Tech Stack](#-tech-stack)
4. [Pipeline Architecture](#-pipeline-architecture)
5. [Repository Structure](#-repository-structure)
6. [Setup & Execution Guide](#-setup--execution-guide)
7. [Model Performance Summary](#-model-performance-summary)
8. [Key Insights & Business Recommendations](#-key-insights--business-recommendations)
9. [References](#-references)

---

## 🔍 Project Overview

Customer churn — the rate at which subscribers discontinue a service — is one of the most costly and measurable threats facing telecommunications companies. Acquiring a new customer costs five to seven times more than retaining an existing one. This project builds an end-to-end machine learning pipeline to:

- **Predict** which customers are at high risk of churning using supervised classification.
- **Explain** the primary behavioral and contractual drivers behind churn using feature importance analysis.
- **Recommend** data-driven retention strategies grounded in the model's findings.

Two models are trained and rigorously compared: a **Logistic Regression** baseline and a **Random Forest Classifier** as the advanced model. Both use `class_weight='balanced'` to handle the inherent class imbalance (~26% churn rate) without data augmentation.

---

## 📦 Dataset Overview

| Property | Detail |
|---|---|
| **Name** | IBM Telco Customer Churn |
| **Source** | [Kaggle — yeanzc/telco-customer-churn-ibm-dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Columns** | 21 features + 1 target (`Churn`) |
| **Churn Rate** | ~26.5% (class imbalance present) |

### Feature Categories

| Category | Features |
|---|---|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Charges** | `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` (Yes / No) |

> **Known Data Quality Issue:** The `TotalCharges` column contains whitespace strings (`" "`) for new customers with `tenure = 0`. These are coerced to `NaN` during ingestion and imputed with the column median.

---

## 🛠 Tech Stack

| Layer | Library | Version |
|---|---|---|
| Data Manipulation | `pandas` | 2.2.2 |
| Numerical Computing | `numpy` | 1.26.4 |
| Machine Learning | `scikit-learn` | 1.5.0 |
| Visualisation | `matplotlib` | 3.9.0 |
| Statistical Plots | `seaborn` | 0.13.2 |
| Notebook Runtime | `jupyter` / `notebook` | 1.0.0 / 7.2.0 |
| Python Kernel | `ipykernel` | 6.29.4 |

---

## 🏗 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    END-TO-END ML PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [1] DATA INGESTION                                                     │
│       └─ pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")           │
│                                                                         │
│  [2] DATA QUALITY & IMPUTATION                                          │
│       ├─ TotalCharges: pd.to_numeric(errors='coerce')                   │
│       └─ NaN → median imputation (affects tenure=0 rows)               │
│                                                                         │
│  [3] EXPLORATORY DATA ANALYSIS                                          │
│       ├─ Churn distribution (count + pie)                               │
│       ├─ Tenure KDE histogram by churn status                           │
│       ├─ Monthly charges boxplot + strip                                │
│       ├─ Contract type vs churn rate                                    │
│       └─ Numeric correlation heatmap                                    │
│                                                                         │
│  [4] FEATURE ENGINEERING & ENCODING                                     │
│       ├─ Drop: customerID                                               │
│       ├─ Target: Churn → {0, 1}                                         │
│       ├─ Binary columns (2 unique values) → LabelEncoder               │
│       ├─ Multi-class columns (>2 unique values) → pd.get_dummies       │
│       └─ Numeric features → StandardScaler (zero-mean, unit-variance)  │
│                                                                         │
│  [5] STRATIFIED TRAIN-TEST SPLIT (80 : 20, random_state=42)            │
│       └─ stratify=y preserves 26.5% churn rate in both sets            │
│                                                                         │
│  [6] DUAL-MODEL TRAINING                                                │
│       ├─ Logistic Regression  (lbfgs, max_iter=1000, balanced)         │
│       └─ Random Forest        (300 trees, balanced, n_jobs=-1)         │
│                                                                         │
│  [7] EVALUATION                                                         │
│       ├─ Accuracy · Precision · Recall · F1-Score · ROC-AUC            │
│       ├─ Confusion Matrices (side-by-side)                              │
│       └─ Overlaid ROC Curves                                            │
│                                                                         │
│  [8] FEATURE IMPORTANCE                                                 │
│       ├─ Random Forest: Gini impurity-based importances                 │
│       └─ Logistic Regression: |coefficient| magnitudes                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Raw dataset (place in root)
├── Vedant_TelcoCustomerChurn.ipynb        ← Main analysis notebook (8 cells)
├── Vedant_ProjectReport.docx              ← Full academic report
├── requirements.txt                       ← Pinned Python dependencies
└── README.md                              ← This file
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.9 or higher installed
- The dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` placed in the project root directory

### Step 1 — Clone or Download the Repository

```bash
# If using git
git clone <your-repository-url>
cd telco-churn-prediction

# Or simply unzip the project folder and navigate into it
```

### Step 2 — Create and Activate a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `Vedant_TelcoCustomerChurn.ipynb` in the browser tab that appears and execute cells sequentially using **Shift + Enter**, or run all at once via **Kernel → Restart & Run All**.

---

## 📊 Model Performance Summary

> Results on the held-out stratified 20% test set (1,409 customers).

| Metric | Logistic Regression | Random Forest |
|---|:---:|:---:|
| **Accuracy** | ~80% | ~79% |
| **Precision** | ~65% | ~63% |
| **Recall** | ~56% | ~74% |
| **F1-Score** | ~60% | ~68% |
| **ROC-AUC** | ~85% | ~84% |

> **Interpretation:** Logistic Regression achieves slightly higher accuracy and ROC-AUC, while Random Forest achieves significantly higher **Recall** — meaning it catches more true churners. In a business context where missing a churner is costlier than a false alarm, **Random Forest is the preferred production model** due to its superior recall and interpretable feature importances.

*Exact values will be printed when the notebook is executed against your environment.*

---

## 💡 Key Insights & Business Recommendations

### Top Churn Drivers (from Random Forest Feature Importance)

| Rank | Feature | Business Interpretation |
|---|---|---|
| 1 | `tenure` | Customers in first 12 months are the highest churn risk |
| 2 | `MonthlyCharges` | Bills above ~$65/month correlate sharply with churn |
| 3 | `TotalCharges` | Low total charges = new customer = high risk |
| 4 | `Contract_Two year` | Long-term contracts dramatically reduce churn |
| 5 | `InternetService_Fiber optic` | Fibre customers churn more — likely unmet speed expectations |
| 6 | `TechSupport_No` | Absence of tech support doubles churn probability |
| 7 | `OnlineSecurity_No` | Customers without security add-on are more likely to leave |

### Strategic Business Recommendations

**1. Early-Tenure Intervention Program**  
Deploy an automated 30-60-90 day onboarding journey for all new customers. Proactive outreach in the first quarter reduces the 3× elevated churn risk observed for customers with `tenure < 12` months.

**2. Contract Conversion Incentives**  
Offer targeted discounts (e.g., one month free) to month-to-month subscribers who convert to annual or two-year contracts. The feature importance analysis shows contract type as one of the strongest protective factors against churn.

**3. Monthly Charge Threshold Alerts**  
Flag customers whose `MonthlyCharges` exceeds $65 for proactive retention calls or loyalty plan offers. The EDA confirms this as a critical inflection point where churn probability rises sharply.

**4. Bundle Add-On Promotions**  
Customers without `TechSupport` or `OnlineSecurity` show significantly elevated churn. Offer these services as free 3-month trials to at-risk segments — converting trial users to paid subscribers while simultaneously reducing churn.

---

## 📚 References

1. IBM. (2019). *Telco Customer Churn Dataset*. Kaggle. https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. Cox, D. R. (1958). The Regression Analysis of Binary Sequences. *Journal of the Royal Statistical Society*, Series B, 20(2), 215–242.
5. Verbeke, W., et al. (2012). New Insights into Churn Prediction in the Telecommunication Sector. *European Journal of Operational Research*, 218(1), 211–229.

---

<div align="center">
  <sub>IBM SkillsBuild Data Analytics with AI Academic Internship &nbsp;|&nbsp; BharatCares × AICTE &nbsp;|&nbsp; Candidate: Vedant</sub>
</div>

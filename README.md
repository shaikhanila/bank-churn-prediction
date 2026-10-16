# Bank Customer Churn Prediction 🏦📊

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?style=flat&logo=jupyter)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg?style=flat)](https://xgboost.readthedocs.io/)

Predicting customer attrition (churn) in the retail banking sector using statistical analysis, exploratory data visualization, and machine learning classification algorithms.

---

## 📌 Project Overview

Customer retention is critical for financial institutions. Acquiring new customers typically costs significantly more than retaining existing ones. This project analyzes customer behavior and banking relationships to predict whether a customer will leave the bank (`Exited = 1`) or stay (`Exited = 0`), enabling banks to proactively target high-risk customers with customized retention campaigns.

---

## 📂 Repository Structure

```text
bank-churn-prediction/
├── bank_churn_prediction.ipynb   # Complete pipeline: EDA, feature engineering, modeling & tuning
├── Bank_churn.ipynb              # Deep-dive analysis: Statistical EDA, PCA, and classification models
├── Bank_churn_.ipynb             # Univariate & bivariate visualizations and baseline experiments
├── .gitignore                    # Ignored artifacts, virtual environments, and cache files
└── README.md                     # Comprehensive project documentation
```

---

## 📊 Dataset Features

The dataset comprises customer demographics, account information, and behavioral indicators:

| Feature | Type | Description |
| :--- | :--- | :--- |
| **CustomerId** | Numerical | Unique identifier for each bank customer |
| **Surname** | Categorical | Customer last name |
| **CreditScore** | Numerical | Credit rating of the customer |
| **Geography** | Categorical | Country of customer residence (France, Spain, Germany) |
| **Gender** | Categorical | Male / Female |
| **Age** | Numerical | Customer age in years |
| **Tenure** | Numerical | Number of years the customer has been with the bank |
| **Balance** | Numerical | Account balance held by the customer |
| **NumOfProducts** | Numerical | Number of bank products used (e.g., credit card, savings, loan) |
| **HasCrCard** | Binary | Whether customer holds a credit card (1 = Yes, 0 = No) |
| **IsActiveMember** | Binary | Whether customer is an active member (1 = Yes, 0 = No) |
| **EstimatedSalary** | Numerical | Estimated annual income |
| **Exited** | Binary (Target)| Churn indicator (1 = Churned / Left bank, 0 = Retained) |

---

## 🛠️ Methodology & Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA):**
   - Missing value verification & data integrity checks.
   - Univariate distributions (age distributions, churn proportions, geography distribution).
   - Bivariate & correlation analysis identifying key churn drivers (e.g., active membership status, age, number of products).

2. **Feature Engineering & Preprocessing:**
   - One-Hot Encoding for categorical variables (`Geography`, `Gender`).
   - Feature scaling with `RobustScaler` and `StandardScaler` to handle outliers and stabilize gradients.
   - Dimensionality reduction and variance examination using **Principal Component Analysis (PCA)**.

3. **Handling Class Imbalance:**
   - Resampling strategies applied to counter dataset skewness towards non-churners using **SMOTE** (Synthetic Minority Over-sampling Technique) and **RandomUnderSampler**.

4. **Model Development & Comparison:**
   - **Logistic Regression** (custom vectorized implementation + Scikit-Learn baseline)
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **Bagging Classifier**
   - **Support Vector Classifier (SVC)**
   - **XGBoost Classifier (Extreme Gradient Boosting)**

5. **Hyperparameter Tuning & Evaluation:**
   - Model optimization via `GridSearchCV` and `RandomizedSearchCV`.
   - Comprehensive performance metrics evaluated: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with Jupyter Notebook or JupyterLab:

```bash
python --version
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shaikhanila/bank-churn-prediction.git
   cd bank-churn-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required libraries:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost notebook
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `bank_churn_prediction.ipynb` to view the analysis and run the cells.

---

## 📈 Key Insights

- **Age Factor:** Older customers demonstrate a higher tendency to churn compared to younger cohorts.
- **Product Engagement:** Customers using 3 or 4 products experience high churn rates, whereas 2 products correlate with optimal retention.
- **Geography:** Regional variations show higher churn rates in specific regions (e.g., Germany) relative to customer volume.
- **Activity Status:** Active members (`IsActiveMember = 1`) show significantly lower churn probability.

---

## 👤 Author

- **Shaikh Anila**
- GitHub: [@shaikhanila](https://github.com/shaikhanila)

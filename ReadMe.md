**Heart Rate (10‑Year CHD) Prediction Project**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Preprocessing](#data-preprocessing)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Modeling](#modeling)
9. [Threshold Selection & Evaluation](#threshold-selection--evaluation)
10. [Results](#results)
11. [Future Work](#future-work)

---

## Project Overview

This repository contains a complete pipeline for predicting the 10-year risk of Coronary Heart Disease (CHD) using the Framingham dataset. The notebook walks through:

* Loading and inspecting the data
* Imputing missing values with median values
* Conducting exploratory data analysis (EDA) to understand distributions and correlations
* Building a logistic regression model
* Optimizing decision thresholds (F1‐maximizing and recall‐targeted)
* Evaluating model performance using accuracy, precision, recall, F1, ROC AUC, and PR AUC

By the end, you will have a clear understanding of each step—from data cleaning through threshold selection—and a baseline logistic regression model.

---

## Dataset

* **Name:** Framingham Heart Study (specifically, the subset of features used to predict 10‑year CHD risk)

* **Columns (features + target):**

  1. `male` (0=female, 1=male)
  2. `age` (in years)
  3. `education` (1–4 levels)
  4. `currentSmoker` (0=no, 1=yes)
  5. `cigsPerDay` (number of cigarettes smoked per day)
  6. `BPMeds` (0=not on blood pressure medication, 1=on medication)
  7. `prevalentStroke` (0=no, 1=yes)
  8. `prevalentHyp` (0=no hypertension, 1=hypertension)
  9. `diabetes` (0=no, 1=yes)
  10. `totChol` (total cholesterol in mg/dL)
  11. `sysBP` (systolic blood pressure in mmHg)
  12. `diaBP` (diastolic blood pressure in mmHg)
  13. `BMI` (Body Mass Index)
  14. `heartRate` (resting heart rate in bpm)
  15. `glucose` (blood glucose in mg/dL)
  16. `TenYearCHD` (target; 0=no CHD in 10 years, 1=CHD within 10 years)

* **Source:** Publicly available Framingham dataset (often provided as `framingham.csv`)

---

## Prerequisites

Before running the notebook, ensure you have the following installed:

1. **Python 3.7+**
2. **pip** or **conda** for package management

**Python packages**:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* (Optional, for advanced steps) `imbalanced-learn` if you wish to experiment with SMOTE or other resampling

---

## Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/RahatalDnyanda/heart-rate-prediction.git
   cd heart-rate-prediction
   ```

2. **Ensure your environment has the required packages**

   * Create a virtual environment (recommended):

     ```bash
     python -m venv venv
     source venv/bin/activate       # On Linux/Mac
     venv\Scripts\activate.bat      # On Windows
     ```
   * Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```
   * Alternatively, install packages one by one as shown in the [Prerequisites](#prerequisites) section.

3. **Download or place `framingham.csv`** in the project’s root directory (same level as the Jupyter notebook).

---

## Usage

1. **Open the Jupyter Notebook**

   ```
   jupyter notebook
   ```

   Then select `heart_rate_chd_prediction.ipynb` from the file browser.

2. **Run All Cells** in order. Each section is commented with a header describing its purpose.

3. **Review Outputs**:

   * Dataframe previews, missing‐value imputation logs, descriptive statistics
   * Distribution plots for continuous variables
   * Correlation heatmap visualizing Pearson correlation coefficients
   * Model training logs and evaluation metrics (accuracy, confusion matrix, classification report)
   * ROC Curve and PR Curve insights
   * Tables summarizing threshold‐based performance (F1‐max, recall ≥ 0.75, default)

## Data Preprocessing

1. **Load Dataset**

   ```python
   import pandas as pd
   df = pd.read_csv("framingham.csv")
   ```

2. **Check for Missing Values**

   ```python
   df.isnull().sum()
   ```

   Columns with missing data (e.g., `education`, `cigsPerDay`, `BPMeds`, `totChol`, `BMI`, `heartRate`, `glucose`) are identified.

3. **Impute Missing Values with Median**

   ```python
   cols_with_nulls = df.columns[df.isnull().any()].tolist()
   for col in cols_with_nulls:
       median_val = df[col].median(skipna=True)
       df[col].fillna(median_val, inplace=True)
   ```

   After this step, there should be **no missing values** in any column.

4. **Verify No Remaining Nulls**

   ```python
   df.isnull().sum()  # All zeros expected
   ```

---

## Exploratory Data Analysis (EDA)

1. **Basic Statistics**

   ```python
   df.describe().T
   ```

   Provides count, mean, standard deviation, min, max, and quartiles for each numeric column.

2. **Univariate Distribution Plots** (Histograms + KDE)

   * Continuous features plotted with `seaborn.histplot(..., kde=True, bins=30)`
   * Features: `['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']`

3. **Bivariate Plots for Binary Features**

   * Countplots of each binary/categorical feature vs. `TenYearCHD`
   * Features: `['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']`

4. **Correlation Heatmap**

   ```python
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt

   corr = df.corr()
   mask = np.triu(np.ones_like(corr, dtype=bool))
   sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm")
   plt.show()
   ```

   Highlights pairwise Pearson correlations among all features and with `TenYearCHD`.

---

## Modeling

### 1. Define Features (`X`) and Target (`y`)

```python
feature_cols = [
    'male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
    'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
    'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
]
X = df[feature_cols]
y = df['TenYearCHD']
```

### 2. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
```

* **Stratified** split ensures the class ratio in train and test sets remains consistent.

### 3. Feature Scaling (Standardization)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

* Standardizes each feature to zero mean and unit variance.
* Scaling is highly recommended for logistic regression.

### 4. Train Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)
```

* Uses L2 penalty by default with `C=1.0`.
* `solver='liblinear'` works well for smaller datasets.

---

## Threshold Selection & Evaluation

### 1. Default Threshold (0.50)

* **Predictions**:

  ```python
  y_proba = model.predict_proba(X_test_scaled)[:, 1]
  y_pred_default = (y_proba >= 0.50).astype(int)
  ```
* **Metrics**:

  ```python
  from sklearn.metrics import (
      accuracy_score, confusion_matrix,
      classification_report, roc_auc_score
  )
  accuracy_score(y_test, y_pred_default)
  confusion_matrix(y_test, y_pred_default)
  classification_report(y_test, y_pred_default, digits=4)
  roc_auc_score(y_test, y_proba)  # AUC unaffected by threshold
  ```
* **Typical Findings**:

  * High overall accuracy (\~ 0.84)
  * Very low positive‐class recall (≈ 5%)
  * ROC AUC around \~ 0.70

### 2. Find “Best‑F1” Threshold

```python
from sklearn.metrics import precision_recall_curve, auc

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = f1_scores.argmax()
best_threshold = thresholds_pr[best_idx]  # ≈ 0.157
```

* **New Predictions**:

  ```python
  y_pred_f1 = (y_proba >= best_threshold).astype(int)
  ```
* **Metrics at Best‑F1**:

  ```python
  accuracy_score(y_test, y_pred_f1)
  precision_score(y_test, y_pred_f1)
  recall_score(y_test, y_pred_f1)
  f1_score(y_test, y_pred_f1)
  confusion_matrix(y_test, y_pred_f1)
  ```

### 3. Find Threshold for Recall ≥ 0.75

```python
target_recall = 0.75
candidates = [
    (thr, p, r)
    for thr, p, r in zip(thresholds_pr, precision[:-1], recall[:-1])
    if r >= target_recall
]
thr_rec75 = max(candidates, key=lambda x: x[1])[0]  # ≈ 0.095
```

* **New Predictions**:

  ```python
  y_pred_rec75 = (y_proba >= thr_rec75).astype(int)
  ```
* **Metrics at Recall≥0.75**:

  ```python
  accuracy_score(y_test, y_pred_rec75)
  precision_score(y_test, y_pred_rec75)
  recall_score(y_test, y_pred_rec75)
  f1_score(y_test, y_pred_rec75)
  confusion_matrix(y_test, y_pred_rec75)
  ```

## Results

* **Default (Threshold = 0.50)**

  * Accuracy ≈ 0.84
  * Precision (CHD=1) ≈ 0.41
  * Recall (CHD=1) ≈ 0.05
  * F1 (CHD=1) ≈ 0.10
  * ROC AUC ≈ 0.70

* **Best‑F1 (Threshold ≈ 0.157)**

  * Accuracy ≈ *(e.g., 0.75–0.80)*
  * Precision (CHD=1) ≈ *(\~0.35–0.45)*
  * Recall (CHD=1) ≈ *(\~0.40–0.50)*
  * F1 (CHD=1) ≈ 0.377

* **Recall ≥ 0.75 (Threshold ≈ 0.095)**

  * Accuracy ≈ *(\~0.50–0.60)*
  * Precision (CHD=1) ≈ *(\~0.20–0.30)*
  * Recall (CHD=1) ≥ 0.75
  * F1 (CHD=1) ≈ *(\~0.30–0.35)*

*(Exact values will vary slightly depending on random seed and data splits. Run the code cells to see your model’s precise performance.)*

---

## Future Work

1. **Hyperparameter Tuning**

   * Use `GridSearchCV` or `RandomizedSearchCV` to optimize `C` and `class_weight` for logistic regression.
   * Evaluate with stratified k‑fold to ensure stability.

2. **Resampling Methods**

   * Try SMOTE (Synthetic Minority Oversampling Technique) to oversample the minority class in training.
   * Experiment with undersampling or combined approaches (e.g., SMOTEENN).

3. **Alternative Models**

   * **Random Forest** or **XGBoost** with `class_weight='balanced'` or `scale_pos_weight` to handle imbalance.
   * Compare ROC AUC and PR AUC across models to identify the best performer.

4. **Feature Engineering**

   * Create interaction terms (e.g., `age × sysBP`, `cigsPerDay × glucose`).
   * Compute polynomial features (e.g., `age²`) if nonlinearity is suspected.
   * Check multicollinearity via VIF and remove or combine highly correlated features.

5. **Probability Calibration**

   * Use `CalibratedClassifierCV` to ensure predicted probabilities match observed frequencies (especially important when thresholds are used for decision‐making).

6. **External Validation**

   * If you can access another CHD dataset, test generalization outside of the Framingham sample.
   * Monitor model drift over time as population statistics change.
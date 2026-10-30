# Bank Marketing Prediction Using Supervised Machine Learning

A college mini project that predicts whether a bank customer will **subscribe to a term deposit** (`yes` / `no`) from customer information and earlier marketing-campaign data. It is a **binary classification** problem solved with three supervised algorithms: Logistic Regression, Decision Tree and Random Forest.

> The model is designed to predict **before** a customer is contacted, so the post-call column `duration` is deliberately excluded (data-leakage prevention, explained in the notebook).

---

## 1. Folder structure

```
Bank_Marketing_Prediction/
├── Bank_Marketing_Prediction.ipynb   # complete notebook (already executed, outputs and graphs included)
├── bank-full.csv                     # ORIGINAL UCI dataset (45,211 rows x 17 columns), comma-separated
├── processed_bank_marketing.csv      # cleaned + feature-engineered data (45,211 rows x 19 columns)
├── requirements.txt
└── README.md
```

`bank-full.csv` is never modified by the notebook (it checks this with a SHA-256 fingerprint at the end). The processed data is written to a separate file.

## 2. Dataset

* **Source:** UCI Machine Learning Repository, *Bank Marketing* (ID 222) - https://archive.ics.uci.edu/dataset/222/bank+marketing
* **Official ZIP:** https://archive.ics.uci.edu/static/public/222/bank+marketing.zip (nested archives; `bank-full.csv` is inside `bank.zip`)
* **Records / columns:** 45,211 rows, 16 input columns + target `y`
* **Target:** `y` - `yes` = subscribed (5,289 customers, 11.70%), `no` = did not subscribe (39,922 customers, 88.30%)
* **Data quality (measured by the notebook):** 0 missing cells, 0 duplicate rows. The string `"unknown"` appears in `poutcome`, `contact`, `education` and `job`; it is kept as a normal category.

### About the delivered `bank-full.csv` (provenance and integrity)
* The original UCI file is **semicolon**-separated. The delivered copy has the **same 45,211 rows and 17 columns and identical values**, saved with **commas** so `pd.read_csv("bank-full.csv")` works directly.
* While this project was being built, the build environment had no access to `archive.ics.uci.edu`, so the original file was obtained from a public GitHub repository that hosts a copy of UCI's `bank-full.csv`. Its **SHA-256 is `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`**, which matches the hash published for the original UCI `bank-full.csv` in an independent open-source loader. The data were then converted only in delimiter/quoting (a round-trip comparison with pandas confirmed the values are equal).
* **You can verify it yourself** (recommended): download the official ZIP, unzip twice (`bank+marketing.zip` -> `bank.zip` -> `bank-full.csv`), and run:
  ```python
  import pandas as pd
  official = pd.read_csv("path/to/official/bank-full.csv", sep=";")   # file extracted from the official UCI ZIP
  delivered = pd.read_csv("bank-full.csv")                            # file delivered in this project
  print(official.equals(delivered))                                   # expected: True
  ```

### Citation
Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306)

Related paper: S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 62:22-31, 2014.

## 3. Installation

Python 3.9 or newer is recommended.

```bash
# 1. open a terminal in the Bank_Marketing_Prediction folder
python -m venv venv

# 2. activate it
#    Windows (PowerShell):  venv\Scripts\Activate.ps1
#    Windows (cmd):         venv\Scripts\activate.bat
#    macOS / Linux:         source venv/bin/activate

# 3. install the libraries
pip install -r requirements.txt
```

No GPU, paid API or API key is needed. The notebook does not download anything; it reads the local `bank-full.csv`.

## 4. How to run the notebook

**VS Code**
1. Install the *Python* and *Jupyter* extensions.
2. Open the `Bank_Marketing_Prediction` folder (`File -> Open Folder`) so the relative file paths work.
3. Open `Bank_Marketing_Prediction.ipynb`, click **Select Kernel** (top right) and choose the `venv` environment.
4. Click **Run All**. It takes about one minute on a normal laptop.

**Jupyter Notebook / JupyterLab**
```bash
jupyter notebook          # or: jupyter lab
```
Open `Bank_Marketing_Prediction.ipynb`, then `Kernel -> Restart & Run All`.

**Command line (optional)**
```bash
jupyter nbconvert --to notebook --execute Bank_Marketing_Prediction.ipynb --output executed.ipynb
```

> The notebook must be run from the folder that contains `bank-full.csv`. Running it overwrites `processed_bank_marketing.csv` with an identical file.

## 5. Notebook outline (25 sections)

1 Project introduction - 2 Problem statement - 3 Dataset source and description - 4 Library imports - 5 Dataset loading - 6 Initial dataset inspection - 7 Data cleaning - 8 Exploratory data analysis - 9 Feature engineering - 10 Train-test split - 11 Data preprocessing pipeline - 12 Logistic Regression - 13 Decision Tree - 14 Random Forest - 15 Model evaluation - 16 Model comparison graphs - 17 Confusion matrices - 18 ROC and precision-recall curves - 19 Cross-validation - 20 Feature importance - 21 Sample prediction - 22 Results and discussion - 23 Conclusion - 24 Limitations - 25 Dataset citation

## 6. Cleaning, feature engineering and preprocessing

**Transformations applied to create `processed_bank_marketing.csv`** (the original file is untouched):

| # | Transformation | Reason |
|---|---|---|
| 1 | Remove exact duplicate rows *only if found* (found: 0) | Avoid double-counting |
| 2 | `y` -> binary (`yes = 1`, `no = 0`) | Numeric target |
| 3 | Drop `duration` | It is the length of the last call, known only *after* the call (**data leakage**) |
| 4 | Add `previously_contacted` = 1 if `pdays != -1` | Gives the sentinel `-1` ("not previously contacted") a clear meaning |
| 5 | Add `days_since_prev_contact` = `pdays` if contacted before, else 0; then drop raw `pdays` | Avoid feeding the sentinel `-1` to the scaler as if it were a number of days |
| 6 | Add `balance_negative` = 1 if `balance < 0` | Overdrawn accounts flag |
| 7 | Add `age_group` (<=24, 25-34, 35-44, 45-54, 55-64, 65+) | Capture non-linear age effects |
| 8 | Keep `"unknown"` values as a category | They are real values, not empty cells |
| 9 | Keep outliers | They are plausible values (large balances, long call sequences) |

**Pipeline (fitted on the training data only):**
* Numerical columns: `SimpleImputer(median)` -> `StandardScaler`
* Categorical columns: `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")`
* Combined with `ColumnTransformer`, wrapped with each model in a `Pipeline`
* Split: 80% train / 20% test, **stratified**, `random_state=42`

## 7. Algorithms

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=50`, `class_weight="balanced"` |
| Random Forest | `n_estimators=300`, `max_depth=12`, `min_samples_leaf=10`, `class_weight="balanced_subsample"` |

All use `random_state=42`. Hyper-parameters were set in advance and **not tuned on the test set**. The final model is selected by **5-fold stratified cross-validation ROC-AUC on the training data**.

## 8. Evaluation metrics

Accuracy, precision, recall, F1-score (for the `yes` class, threshold 0.5), ROC-AUC, average precision, classification reports, confusion matrices, ROC curves, precision-recall curves, and 5-fold stratified cross-validation. Because 88% of customers say "no", accuracy alone is misleading (a model that always predicts "no" gets 88.3% accuracy but finds no subscribers).

## 9. Results from the verified execution

These values were produced by executing the notebook (seed 42) and are the same numbers stored in the notebook's outputs. They were identical in two different environments (scikit-learn 1.8 / pandas 3.0 and scikit-learn 1.5.2 / pandas 2.2.3). Other library versions may differ slightly; the notebook always prints the actual values.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7656 | 0.2775 | 0.6257 | 0.3844 | 0.7740 | 0.4112 |
| Decision Tree | 0.8232 | 0.3428 | 0.5577 | 0.4246 | 0.7566 | 0.3551 |
| Random Forest | 0.8182 | 0.3430 | 0.6049 | 0.4378 | 0.7976 | 0.4417 |

Random Forest was selected (highest cross-validated ROC-AUC, 0.788). In an optional experiment, adding the leaky column `duration` raised Logistic Regression's ROC-AUC from 0.774 to 0.909, which is why it is excluded.

## 10. Graphs generated (all with titles, axis labels and legends)

Target distribution - age distribution - numerical distributions - boxplots (outliers) - correlation heatmap - subscription by job - subscription rate by month - model comparison bar chart - confusion matrices (3) - ROC curves - precision-recall curves - cross-validation comparison - Random Forest feature importance (encoded and grouped).

## 11. Limitations (short)

Single bank and period (about 2008-2010); results are associative, not causal; the default 0.5 threshold ignores call costs; limited hyper-parameter search; one random split; fairness and privacy considerations apply to using attributes such as age or marital status in real marketing. See Section 24 of the notebook for the full list.

## 12. Reproducibility checklist

- [x] Fixed random seed (`42`) for the split, models and cross-validation
- [x] Relative paths only, no downloads during normal execution
- [x] No GPU, API keys or paid services
- [x] Original dataset checked unchanged via SHA-256 at the end of the notebook
#   M L _ m i n i _ p r o j e c t  
 
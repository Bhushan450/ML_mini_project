# Bank Marketing Prediction (Simplified)

## Objective
Predict whether a bank customer will **subscribe to a term deposit** (`y` = yes / no) using supervised machine learning, and compare three algorithms on the same data. This is a binary classification mini project focused on cleaning, preprocessing, training, evaluation and comparison.

## Dataset
* **UCI Bank Marketing** (`bank-full.csv`): 45,211 customers, 16 input columns + target `y`.
  Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
* About 11.7% of customers subscribed, so the classes are **imbalanced**.
* The file is stored with comma separators (the original UCI file uses semicolons); the values are unchanged. The notebook only **reads** it and never modifies it.
* Citation: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## Project files
```
Bank_Marketing_Prediction/
├── Bank_Marketing_Prediction.ipynb   # the whole project (already executed)
├── bank-full.csv                     # original dataset (unchanged)
├── processed_bank_marketing.csv      # data after cleaning (duration removed, y as 0/1), created by the notebook
├── requirements.txt
└── README.md
```

## What the notebook does
1. **Import libraries and load dataset:** shows the first five rows, shape and column names.
2. **Data cleaning:** checks missing values, duplicate rows, data types and invalid values (age, day, duration, campaign, previous, pdays, target). Fixes are applied only if a problem is found; the notebook prints a summary of what was actually done. In this dataset no missing values, duplicates or invalid values exist, so nothing was removed. The text value `"unknown"` is kept as a normal category, and `pdays = -1` ("not previously contacted") is kept as is.
3. **Data preprocessing:**
   * `y` is converted to 1 (yes) / 0 (no) and separated from the inputs.
   * `duration` is **excluded**: it is the call length, known only after the call, so using it would be data leakage when predicting before contacting a customer.
   * Numerical features: median imputation + `StandardScaler`. Categorical features: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`. Combined with `ColumnTransformer`.
   * 80/20 train-test split with `stratify=y` and `random_state=42`.
   * Preprocessing sits inside each model `Pipeline`, so it is fitted on the training data only.
   * The cleaned data (15 input columns + binary `y`, without `duration`) is saved to `processed_bank_marketing.csv`. Scaling and one-hot encoding are not saved in this file because they happen inside the pipeline, after the split, to avoid leakage.
   * Class imbalance is handled the same way in all models with `class_weight="balanced"`.
4. **Model training:** Logistic Regression, Decision Tree (`max_depth=6`, `min_samples_leaf=50`) and Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_leaf=10`), all on the same training split.
5. **Model evaluation:** Accuracy, Precision, Recall, F1-score (on the test set, using predicted labels) and ROC-AUC (using predicted probabilities).
6. **Model comparison:** one table sorted by ROC-AUC (the primary selection metric), the best model, and an explanation of why accuracy alone can mislead.

## Evaluation metrics (in short)
* **Accuracy:** share of all predictions that are correct.
* **Precision:** of customers predicted "yes", how many really subscribed.
* **Recall:** of customers who really subscribed, how many the model found.
* **F1-score:** balance of precision and recall.
* **ROC-AUC:** how well the model ranks likely subscribers above unlikely ones (0.5 = guessing).

Because most customers say "no", always predicting "no" already gives about 88% accuracy while finding nobody, so accuracy alone is not enough.

## Results (actual run, seed 42)
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8199 | 0.3468 | 0.6106 | 0.4423 | 0.7993 |
| Logistic Regression | 0.7548 | 0.2662 | 0.6238 | 0.3732 | 0.7722 |
| Decision Tree | 0.8232 | 0.3428 | 0.5577 | 0.4246 | 0.7566 |

Best model by ROC-AUC: **Random Forest**. (The notebook prints the actual numbers each time it runs; other library versions may differ slightly.)

## How to run
```bash
# in the project folder
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
jupyter notebook               # open Bank_Marketing_Prediction.ipynb, then Run All
```
In **VS Code**: open the folder, open the notebook, select the `venv` kernel, click **Run All**. Keep the notebook in the same folder as `bank-full.csv`. It takes a few seconds.

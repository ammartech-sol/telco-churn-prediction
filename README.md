# Telco Customer Churn Prediction

A machine learning project to predict customer churn using the Telco Customer Churn dataset from Kaggle. The goal is to identify customers likely to leave before they actually do, which is more valuable to a business than simply knowing the overall churn rate.

\---

## Dataset

* **Source:** Kaggle - Telco Customer Churn
* **Size:** 7,043 customers, 21 features
* **Target:** Churn (Yes/No) - about 26.5% of customers churned

\---

## Project Walkthrough

### 1\. Data Cleaning

* Dropped `customerID` since it's just a unique identifier with no predictive value
* `TotalCharges` was stored as an object column due to whitespace entries - converted it to float and filled 11 missing values with 0 (these were customers with zero tenure, so they hadn't been billed yet)
* No duplicate rows were found

### 2\. Feature Engineering

Three new features were created based on domain logic:

* **`is\\\\\\\_month\\\\\\\_to\\\\\\\_month`** - Binary flag for customers on a month-to-month contract, since they are significantly more likely to churn than customers locked into 1 or 2 year contracts
* **`avg\\\\\\\_monthly\\\\\\\_spend`** - `TotalCharges / (tenure + 1)`, a smoother version of monthly spend that avoids division by zero for brand new customers
* **`total\\\\\\\_services`** - Sum of all active services per customer. The logic here is that a customer using 6 services is deeply embedded and has more friction to leave, while a customer using 1 service can walk away easily

### 3\. Encoding

* Binary columns (gender, Partner, Dependents, etc.) were mapped to 0/1 directly
* Columns with a third value like "No internet service" or "No phone service" were treated the same as "No" - they both mean the customer doesn't have that feature
* `InternetService`, `Contract`, and `PaymentMethod` were one-hot encoded using `ColumnTransformer`
* The encoder was fit only on training data to avoid data leakage

\---

## Modeling

### Baseline: Random Forest

The default RandomForestClassifier was trained first to get a baseline.

```
Train score: 0.998
Test score:  0.783
```

The massive gap between train and test score signals overfitting. Cross-validation confirmed this:

```
Average Recall:  0.49
Average F1:      0.55
Average ROC-AUC: 0.83
```

### Why accuracy doesn't matter here

With 73.5% of customers being non-churners, a model that predicts "No churn" every single time would get 73.5% accuracy without learning anything. Accuracy is a misleading metric for imbalanced problems like this. The metrics that actually matter are **Recall** (catching real churners) and **ROC-AUC** (overall ranking ability).

### class\_weight='balanced' made things worse

One common approach for imbalanced data is to set `class\\\\\\\_weight='balanced'`, which forces the model to penalize misclassifying the minority class more heavily. It was tested here - and it actually reduced recall on the churn class compared to the default model. This happens because the dataset isn't severely imbalanced (26% is still a meaningful minority), and forcing aggressive reweighting can push the model to overcorrect in ways that hurt overall discrimination. It was dropped after testing.

\---

## Hyperparameter Tuning

`RandomizedSearchCV` was run twice - once optimizing for F1 and once for Recall - across 30 random combinations with 5-fold cross-validation.

||F1-Optimized|Recall-Optimized|
|-|-|-|
|Best CV Score|0.571|0.518|
|n\_estimators|300|400|
|max\_depth|10|15|
|bootstrap|True|False|

Both tuned models performed similarly on the test set, with only marginal improvement over the baseline. Tuning alone wasn't the answer.

\---

## Threshold Tuning

The real breakthrough came from adjusting the classification threshold. By default, scikit-learn classifies a customer as a churner if predicted probability >= 0.5. Lowering this threshold means the model flags customers earlier, catching more real churners at the cost of more false alarms.

The optimal threshold was found by sweeping the precision-recall curve and maximizing F1:

**Best Threshold: 0.323**

### Final Results (F1-Optimized Model + Threshold Tuning)

|Metric|Default (0.5)|After Threshold Tuning (0.323)|
|-|-|-|
|Recall (Churn)|0.52|0.76|
|Precision (Churn)|0.63|0.55|
|F1 (Churn)|0.57|0.63|
|ROC-AUC|0.83|-|

Recall jumped from 0.52 to 0.76. In a churn context, missing a real churner is more costly than a false alarm - a wrongly flagged customer just gets a retention offer they didn't need, but a missed churner is lost revenue.

\---

## Saved Files

|File|Description|
|-|-|
|`churn\\\\\\\_model.pkl`|Trained Random Forest (F1-optimized params)|
|`encoder.pkl`|Fitted ColumnTransformer for preprocessing new data|
|`telco\\\\\\\_churn.ipynb`|Full notebook with code and outputs|

\---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

\---

## How to Run

1. Clone the repo
2. The dataset is already included in the data/ folder.
3. &#x20;  Just make sure your notebook is pointing to the correct path: data/Telco-Customer-Churn.csv
4. Open and run `analysis.ipynb` from top to bottom

\---

## Key Takeaways

* Accuracy is useless as the primary metric on imbalanced classification problems
* `class\\\\\\\_weight='balanced'` is not a guaranteed fix - always test it rather than assuming it helps
* Hyperparameter tuning gave marginal gains here; threshold tuning gave much larger ones
* Data leakage in preprocessing is easy to get wrong - always fit the encoder on training data only and transform test separately
* For churn, recall is the metric to optimize for because missing a real churner costs more than a false alarm


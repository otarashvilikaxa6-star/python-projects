# Customer Churn Prediction

## About This Project

In this project I built a model to predict which telecom customers are likely to cancel their service (churn). I worked with the Telco Customer Churn dataset: 7,043 customers and 21 columns covering demographics, services, contract type, and billing. About 26.5% of the customers in the data churned.

The work is done in Python in a Jupyter notebook, using pandas, scikit-learn, XGBoost, CatBoost, and SHAP.

## How I Built It

1. **Cleaning:** I fixed 11 blank `TotalCharges` values (customers with zero tenure) by setting them to 0, converted the column to numeric, and encoded the yes/no and categorical columns as numbers.
2. **Exploration:** I charted the distribution of each variable and compared churn across groups. For example, 41.7% of senior citizens churned versus 23.6% of other customers, and churned customers had a median tenure of 10 months versus 38 months for customers who stayed. I also checked correlations between features and fitted a small decision tree to see the main splits.
3. **Model comparison:** After an 80/20 train/test split and feature scaling, I compared seven algorithms with 5-fold cross-validation: logistic regression, SVC, KNN, Gaussian Naive Bayes, decision tree, random forest, and gradient boosting.
4. **Evaluation:** I fitted logistic regression, gradient boosting, KNN, XGBoost, and CatBoost and compared them on the test set using AUC, accuracy, precision, recall, F1, and F2, plus a lift analysis.
5. **Feature importance and tuning:** The gradient boosting model ranked contract type (42%), tenure (17%), monthly charges (12%), and online security (10%) as the most important features. I retrained on those four features, tuned the model with a 72-combination grid search using 5-fold cross-validation, and used SHAP to explain predictions.

## Results (test set)

| Model | AUC | Recall |
|---|---|---|
| Logistic regression | 0.848 | 0.826 |
| Gradient boosting | 0.848 | 0.522 |
| CatBoost | 0.847 | 0.562 |
| KNN | 0.827 | 0.528 |
| XGBoost | 0.825 | 0.626 |

Gradient boosting on the four selected features reached a test AUC of about 0.840.

## Tools and Skills Used

- Python, pandas, NumPy, Matplotlib, seaborn
- scikit-learn, XGBoost, CatBoost
- Cross-validation, grid search, lift analysis, SHAP

**File:** [Churn_Prediction.ipynb](Churn_Prediction.ipynb)

# Credit Card Fraud Detection

## About This Project

In this project I built models to detect fraudulent credit card transactions. I used the 2023 credit card fraud detection dataset from Kaggle, which has 568,630 transactions with 28 anonymized features (V1 to V28), the transaction amount, and a fraud label. The two classes are balanced (50% fraud, 50% not fraud).

The work is done in Python in a Jupyter notebook, using pandas, scikit-learn, and Matplotlib.

## How I Built It

1. **Exploration:** I checked each column for missing values and summarized and plotted each variable (statistics and box plots).
2. **Feature reduction:** I calculated the correlations between features and with the target, then removed 17 highly correlated features. This left 11 anonymized features plus the transaction amount, with no highly correlated pairs.
3. **Data split:** I set aside a random 10% of the data as a separate test file, then split the rest 80/20 into training and validation sets and standardized the features.
4. **Model comparison:** I compared eight algorithms with 5-fold cross-validation: logistic regression, linear SVC, kernel SVM, KNN, Gaussian Naive Bayes, decision tree, random forest, and gradient boosting. I then tuned the number of neighbors for KNN and the number of trees for random forest and gradient boosting.
5. **Evaluation:** I compared KNN, gradient boosting, and random forest on the validation set using AUC, accuracy, precision, recall, F1, and F2, and ran a lift analysis.
6. **Feature importance and tuning:** I ranked features by importance in the random forest and gradient boosting models, kept the most important ones, and tuned gradient boosting with a 36-combination grid search using 5-fold cross-validation.

## Results (validation set, all remaining features)

| Model | AUC | Precision | Recall |
|---|---|---|---|
| KNN | 0.902 | 0.861 | 0.740 |
| Gradient boosting | 0.891 | 0.891 | 0.684 |
| Random forest | 0.840 | 0.909 | 0.501 |

After keeping only the most important features, gradient boosting reached a validation AUC of about 0.816 (precision 0.858, recall 0.593).

## Tools and Skills Used

- Python, pandas, NumPy, Matplotlib, seaborn
- scikit-learn classification models and metrics
- Correlation-based feature selection, cross-validation, grid search, lift analysis

**File:** [Credit_Card_Fraud.ipynb](Credit_Card_Fraud.ipynb)

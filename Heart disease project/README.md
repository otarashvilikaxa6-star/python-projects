# Heart Disease Prediction

## About This Project

In this project I built models that predict whether a patient has heart disease from clinical measurements. The dataset has 303 patients and 13 features (such as age, sex, chest pain type, resting blood pressure, cholesterol, and maximum heart rate) plus a target column. There are no missing values.

The work is done in Python in a Jupyter notebook, using pandas, scikit-learn, and Matplotlib.

## How I Built It

1. **Exploration:** I checked for missing values, compared the target by sex (the target is 1 for 75% of the female patients and 45% of the male patients), and plotted a correlation heatmap.
2. **Baseline models:** I split the data 80/20 and compared three classifiers: logistic regression, KNN, and random forest.
3. **KNN tuning:** I tested 1 to 20 neighbors and plotted training and test accuracy to see how the number of neighbors affected the score.
4. **Hyperparameter tuning:** I used `RandomizedSearchCV` with 5-fold cross-validation (10 candidates each) on logistic regression and random forest.

## Results (test set of 61 patients)

| Model | Accuracy |
|---|---|
| Logistic regression | 91.8% |
| Random forest | 90.2% |
| KNN (default) | 57.4% |
| KNN (best of 1-20 neighbors) | 62.3% |
| Logistic regression (tuned) | 90.2% |
| Random forest (tuned) | 83.6% |

The test set is small and the split has no fixed random seed, so these figures change from run to run.

## Tools and Skills Used

- Python, pandas, NumPy, Matplotlib
- scikit-learn: logistic regression, KNN, random forest, `RandomizedSearchCV`
- Cross-validation, hyperparameter tuning, model comparison

**File:** [heart_disease_project.ipynb](heart_disease_project.ipynb)

# Python Projects

Welcome to my Python portfolio. This repository contains five machine learning projects that show how I work with data in Python, from cleaning and exploration to modeling and evaluation. They cover classification, regression, and deep learning on image data.

## Projects

### churn prediction

Predicting which of 7,043 telecom customers will cancel their service. I explored the data, compared seven algorithms with cross-validation, evaluated five models (including XGBoost and CatBoost), identified contract type and tenure as the strongest predictors, and explained predictions with SHAP.

### credit card fraud detection

Detecting fraudulent transactions in a dataset of 568,630 credit card transactions. I removed highly correlated features, compared eight algorithms, evaluated KNN, gradient boosting, and random forest with AUC, precision, and recall, and tuned the final model with a grid search.

### heart disease prediction

Predicting heart disease from clinical data on 303 patients. I compared logistic regression, KNN, and random forest, and tuned the models with randomized search and cross-validation; logistic regression reached 91.8% test accuracy.

### dog vision project

An image classifier for 120 dog breeds, trained on 10,222 labeled photos. I used transfer learning with a pretrained MobileNetV2 model from TensorFlow Hub, built with TensorFlow and Keras.

### bulldozer price prediction

A regression model that predicts the auction price of used bulldozers from 412,698 historical sales. I engineered date features, handled missing values, validated on a time-based split, and compared Random Forest models; the final model reached an R² of 0.883.

## Skills Demonstrated

- **Machine learning:** scikit-learn, XGBoost, CatBoost, Random Forest, gradient boosting, model evaluation (AUC, precision, recall, RMSLE, MAE, R²)
- **Deep learning:** TensorFlow 2, Keras, TensorFlow Hub, transfer learning, TensorBoard
- **Data preparation:** pandas, NumPy, feature engineering, missing-value handling, feature selection
- **Model validation and tuning:** cross-validation, grid search, randomized search, lift analysis, SHAP
- **Visualization:** Matplotlib, seaborn
- **Tools:** Jupyter, Google Colab

Each project folder has its own README with more detail and the notebook.

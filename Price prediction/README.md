# Bulldozer Sale Price Prediction

## About This Project

In this project I built a machine learning model that predicts the auction sale price of used bulldozers from their attributes. I worked with the Blue Book for Bulldozers dataset, which has 412,698 historical sales records covering sales from 1989 to 2012. Each record includes details such as the year the machine was made, usage hours, model, product group, and state of sale.

The work is done in Python in a Jupyter notebook, using pandas, NumPy, Matplotlib, and scikit-learn.

## How I Built It

1. **Exploration:** I checked missing values across the columns (some, like `UsageBand` and `fiModelSeries`, were missing for most rows), plotted the price distribution, and parsed and sorted the sale dates.
2. **Feature engineering:** I split the sale date into year, month, day, day of week, and day of year.
3. **Data cleaning:** I converted text columns to numeric category codes and filled missing numeric values with the median. For every column with missing values I added a True/False "is missing" column, so the model can still use the fact that a value was absent. After this step the data had 102 features.
4. **Time-based validation:** Because the data is a time series of sales, I trained on all sales before 2012 (401,125 rows) and validated on the 2012 sales (11,573 rows), rather than splitting at random.
5. **Modeling:** I used a Random Forest regressor and compared three versions on the validation set, using RMSLE, mean absolute error (MAE), and R².
6. **Test predictions:** I wrote a preprocessing function that applies the same steps to the test file and generated predictions with the final model.

## Results (validation set, 2012 sales)

| Model | RMSLE | MAE | R² |
|---|---|---|---|
| Baseline (10,000 samples per tree) | 0.294 | $7,177 | 0.832 |
| Randomized search (2 iterations, 5-fold CV) | 0.456 | $11,665 | 0.614 |
| Final Random Forest (40 trees, `max_features=0.5`, `min_samples_split=14`) | **0.245** | **$5,935** | **0.883** |

The randomized search ran only 2 iterations, so it did not improve on the baseline; the final model uses manually chosen settings.

## Tools and Skills Used

- Python, pandas, NumPy, Matplotlib
- scikit-learn: `RandomForestRegressor`, `RandomizedSearchCV`, regression metrics
- Feature engineering, missing-value handling, time-based train/validation split

**File:** [bulldozer-price-prediction.ipynb](bulldozer-price-prediction.ipynb)

# online-svr

A Python library for time series forecasting with **Support Vector Regression**, featuring a scikit-learn compatible API and support for both offline and online (incremental) learning.

## Installation

```bash
pip install online-svr
```

## Key Classes

| Class | Description |
|---|---|
| `LagTransformer` | Identifies statistically significant lags via PACF and builds lag feature matrices |
| `SVRForecaster` | SVR estimator with `fit`, `predict`, and `partial_fit` (online learning) |
| `TimeSeriesSplitter` | Loads a CSV and splits it into train/validation/test sets |

## Quick Start

```python
import numpy as np
from online_svr import LagTransformer, SVRForecaster

# 1. Identify significant lags
transformer = LagTransformer(max_lags=10, significance_level=0.05)
transformer.fit(y)
print("Significant lags:", transformer.lags_)

# 2. Build lag feature matrix
X = transformer.transform(y)
max_lag = max(transformer.lags_)
y_aligned = y[max_lag:]

# 3. Train/test split (sequential — no shuffle)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_aligned[:split], y_aligned[split:]

# 4. Fit the forecaster
model = SVRForecaster()
model.fit(X_train, y_train)

# 5. Offline multi-step prediction
predictions = model.predict(X_test[-1:], steps=5)
print("Offline predictions:", predictions)

# 6. Online update with a true observation
model.partial_fit(X_test[-1:], [y_test[-1]])
updated = model.predict(X_test[-1:], steps=1)
print("After online update:", updated)
```

## Loading data from CSV

```python
from online_svr import TimeSeriesSplitter

splitter = TimeSeriesSplitter(test_size=0.2, val_size=0.5)
train, val, test = splitter.split("data.csv")  # CSV must have 'date' and 'target' columns
```

## How it works

### LagTransformer

1. **fit(X)** — computes the PACF of the series and stores the lags whose partial autocorrelation exceeds the confidence interval at the given `significance_level`.
2. **transform(X)** — builds a 2-D feature matrix where each column is a lagged copy of the series, aligned so every row has all required history.

### SVRForecaster

1. **fit(X, y)** — normalises features with `StandardScaler`, then runs `GridSearchCV` over the `param_grid` to find the best SVR hyperparameters.
2. **predict(X, steps)** — recursive multi-step forecast: each predicted value is appended to the feature window and used to predict the next step.
3. **partial_fit(X, y)** — online learning: refits the estimator on a single new observation to adapt the model as new data arrives.

## Requirements

- Python >= 3.9
- numpy, pandas, scikit-learn, statsmodels, scipy

import numpy as np
from online_svr import LagTransformer, SVRForecaster

# Synthetic trending time series with seasonal component
np.random.seed(42)
n = 60
y = np.array([i * 1.5 + 3 * np.sin(i * 0.4) for i in range(n)])

# 1. Identify significant lags via PACF
transformer = LagTransformer(max_lags=10, significance_level=0.05)
transformer.fit(y)
print("Significant lags:", transformer.lags_)

# 2. Build lag feature matrix
X = transformer.transform(y)
max_lag = max(transformer.lags_)
y_aligned = y[max_lag:]

# 3. Sequential train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_aligned[:split], y_aligned[split:]

# 4. Fit the forecaster
model = SVRForecaster(cv=3)
model.fit(X_train, y_train)
print("Best SVR params:", model.estimator_.get_params())

# 5. Offline multi-step prediction (5 steps ahead from last test observation)
predictions = model.predict(X_test[-1:], steps=5)
print("Offline predictions (5 steps):", np.round(predictions, 2))

# 6. Online update with a new true observation, then predict again
model.partial_fit(X_test[-1:], [y_test[-1]])
updated_pred = model.predict(X_test[-1:], steps=1)
print("Prediction after online update:", np.round(updated_pred, 2))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf
from scipy.stats import norm


class TimeSeriesSplitter:
    """Loads a CSV time series and splits it into train/validation/test sets.

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction reserved for the combined validation+test split.
    val_size : float, default=0.5
        Fraction of the temp split assigned to validation.
    """

    def __init__(self, test_size=0.2, val_size=0.5):
        self.test_size = test_size
        self.val_size = val_size

    def split(self, file_path):
        """Load a CSV and return train, validation and test DataFrames.

        The CSV must contain 'date' and 'target' columns.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        train_data, val_data, test_data : tuple of pd.DataFrame
        """
        data = pd.read_csv(file_path, parse_dates=["date"])
        train_data, temp_data = train_test_split(
            data, test_size=self.test_size, shuffle=False
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=self.val_size, shuffle=False
        )
        return train_data, val_data, test_data


class LagTransformer(BaseEstimator, TransformerMixin):
    """Identifies significant lags via PACF and builds lag feature matrices.

    Parameters
    ----------
    max_lags : int, default=10
        Maximum number of lags to evaluate.
    significance_level : float, default=0.05
        Significance level for the PACF confidence interval.

    Attributes
    ----------
    lags_ : list of int
        Significant lag indices found during fit.
    """

    def __init__(self, max_lags=10, significance_level=0.05):
        self.max_lags = max_lags
        self.significance_level = significance_level

    def fit(self, X, y=None):
        """Identify statistically significant lags in the time series.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Univariate time series values.

        Returns
        -------
        self
        """
        X = np.asarray(X).ravel()
        max_allowed = min(self.max_lags, len(X) // 2 - 1)
        if max_allowed <= 0:
            raise ValueError("Insufficient data points to calculate lags.")

        pacf_values = pacf(X, nlags=max_allowed)
        z = norm.ppf(1 - self.significance_level / 2)
        threshold = z / np.sqrt(len(X))
        self.lags_ = [
            i for i, coef in enumerate(pacf_values)
            if abs(coef) > threshold and i != 0
        ]
        return self

    def transform(self, X):
        """Build a lag feature matrix from the time series.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Univariate time series values.

        Returns
        -------
        X_lag : ndarray of shape (n_samples - max_lag, n_lags)
        """
        X = np.asarray(X).ravel()
        max_lag = max(self.lags_)
        features = np.column_stack(
            [X[max_lag - lag: len(X) - lag] for lag in self.lags_]
        )
        return features


class SVRForecaster(BaseEstimator):
    """SVR-based time series forecaster with offline and online learning.

    Parameters
    ----------
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV. Uses a default grid if None.
    cv : int, default=5
        Number of cross-validation folds.

    Attributes
    ----------
    estimator_ : SVR
        Best SVR estimator found by GridSearchCV.
    scaler_ : StandardScaler
        Scaler fitted on the training features.
    """

    def __init__(self, param_grid=None, cv=5):
        self.cv = cv
        self.param_grid = param_grid or {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1e-3, 1e-2, 0.1, 1],
            "epsilon": [0.1, 0.2, 0.5],
            "kernel": ["linear", "poly", "rbf"],
        }

    def fit(self, X, y):
        """Fit the SVR model using GridSearchCV.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_lags)
            Lag feature matrix (e.g. from LagTransformer).
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = np.asarray(X), np.asarray(y).ravel()
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        grid_search = GridSearchCV(SVR(), self.param_grid, cv=self.cv)
        grid_search.fit(X_scaled, y)
        self.estimator_ = grid_search.best_estimator_
        return self

    def predict(self, X, steps=1):
        """Make multi-step ahead predictions (offline mode).

        Starting from the last row of X, the model recursively predicts
        `steps` values by feeding each prediction back as a feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_lags) or (n_lags,)
            Feature matrix. Only the last row is used as the initial state.
        steps : int, default=1
            Number of steps to forecast ahead.

        Returns
        -------
        predictions : ndarray of shape (steps,)
        """
        X = np.asarray(X)
        last = X[-1].copy() if X.ndim > 1 else X.copy()
        predictions = []
        for _ in range(steps):
            last_scaled = self.scaler_.transform([last])[0]
            pred = self.estimator_.predict([last_scaled])[0]
            predictions.append(pred)
            last = np.roll(last, -1)
            last[-1] = pred
        return np.array(predictions)

    def partial_fit(self, X, y):
        """Update the model with a new observation (online learning).

        Refits the estimator on the provided sample to adapt the model
        incrementally as new data arrives.

        Parameters
        ----------
        X : array-like of shape (1, n_lags) or (n_lags,)
            Feature vector for the new observation.
        y : array-like of shape (1,)
            True target value.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(y).ravel()
        X_scaled = self.scaler_.transform(X)
        self.estimator_.fit(X_scaled, y)
        return self

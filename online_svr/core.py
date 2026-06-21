import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf
from scipy.stats import norm


def _extract_series(X):
    """Return a 1-D float array from X.

    Accepts:
    - 1-D array-like of numeric values
    - pandas Series
    - pandas DataFrame with 1 column (values) or 2 columns (date + values).
      When 2 columns are present the datetime column is used to sort the rows
      before extracting the numeric column.
    - 2-D array with shape (n, 1) or (n, 2).  For (n, 2) the first column is
      treated as an index/date and the second as the target values.
    """
    if isinstance(X, pd.Series):
        return X.values.astype(float)

    if isinstance(X, pd.DataFrame):
        if X.shape[1] == 1:
            return X.iloc[:, 0].values.astype(float)
        if X.shape[1] == 2:
            date_col = None
            for col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    date_col = col
                    break
                try:
                    parsed = pd.to_datetime(X[col], errors="raise")
                    date_col = col
                    X = X.copy()
                    X[col] = parsed
                    break
                except Exception:
                    pass
            if date_col is not None:
                value_col = [c for c in X.columns if c != date_col][0]
                return X.sort_values(date_col)[value_col].values.astype(float)
            # no date detected — use the second column as target
            return X.iloc[:, 1].values.astype(float)
        raise ValueError(
            "DataFrame must have 1 column (values) or 2 columns (date, values)."
        )

    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr.ravel().astype(float)
        if arr.shape[1] == 2:
            return arr[:, 1].astype(float)
        raise ValueError(
            "2-D array must have shape (n, 1) or (n, 2). "
            "For (n, 2) column 0 is treated as the date/index and column 1 as the values."
        )
    raise ValueError("Input must be 1-D or 2-D.")


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
        X : array-like of shape (n_samples,), pandas Series, or DataFrame
            Univariate time series values.  A DataFrame with 2 columns is
            interpreted as (date, value); the date column is used to sort the
            rows before fitting.

        Returns
        -------
        self
        """
        X = _extract_series(X)
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
        X : array-like of shape (n_samples,), pandas Series, or DataFrame
            Univariate time series values.  Accepts the same formats as
            :meth:`fit`.

        Returns
        -------
        X_lag : ndarray of shape (n_samples - max_lag, n_lags)
        """
        X = _extract_series(X)
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
    test_size : float, default=0.2
        Fraction of data held out as test set during fit. Stored in
        ``X_test_`` and ``y_test_``. Ignored when ``one_step_ahead=True``.
    one_step_ahead : bool, default=False
        When True, the model trains on the full series and ``predict``
        always returns a single next-step forecast, ignoring ``steps``.

    Attributes
    ----------
    estimator_ : SVR
        Best SVR estimator found by GridSearchCV.
    scaler_ : StandardScaler
        Scaler fitted on the training features.
    X_test_ : ndarray
        Held-out feature matrix (only when ``one_step_ahead=False``).
    y_test_ : ndarray
        Held-out target values (only when ``one_step_ahead=False``).
    """

    def __init__(self, param_grid=None, cv=5, test_size=0.2, one_step_ahead=False):
        self.cv = cv
        self.test_size = test_size
        self.one_step_ahead = one_step_ahead
        self.param_grid = param_grid or {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1e-3, 1e-2, 0.1, 1],
            "epsilon": [0.1, 0.2, 0.5],
            "kernel": ["linear", "poly", "rbf"],
        }

    def fit(self, X, y):
        """Fit the SVR model using GridSearchCV.

        When ``one_step_ahead=False``, splits the data sequentially using
        ``test_size`` and stores the held-out set in ``X_test_``/``y_test_``.
        When ``one_step_ahead=True``, trains on the full series.

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

        if self.one_step_ahead:
            X_train, y_train = X, y
        else:
            split = int(len(X) * (1 - self.test_size))
            X_train, self.X_test_ = X[:split], X[split:]
            y_train, self.y_test_ = y[:split], y[split:]

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        grid_search = GridSearchCV(SVR(), self.param_grid, cv=self.cv)
        grid_search.fit(X_scaled, y_train)
        self.estimator_ = grid_search.best_estimator_
        return self

    def predict(self, X, steps=1):
        """Make predictions from the last row of X.

        When ``one_step_ahead=True``, always returns a single forecast
        regardless of ``steps``. Otherwise performs recursive multi-step
        prediction, feeding each output back as input for the next step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_lags) or (n_lags,)
            Feature matrix. Only the last row is used as the initial state.
        steps : int, default=1
            Number of steps ahead. Ignored when ``one_step_ahead=True``.

        Returns
        -------
        predictions : ndarray of shape (1,) or (steps,)
        """
        X = np.asarray(X)
        last = X[-1].copy() if X.ndim > 1 else X.copy()
        n_steps = 1 if self.one_step_ahead else steps
        predictions = []
        for _ in range(n_steps):
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

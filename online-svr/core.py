import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import  pacf
import numpy as np
from scipy.stats import norm


class Serie_analisys:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_and_split(self, test_size=0.2, val_size=0.5):
 
        self.data = pd.read_csv(self.file_path, parse_dates=["date"])
        train_data, temp_data = train_test_split(self.data, test_size=test_size, shuffle=False)  
        val_data, test_data = train_test_split(temp_data, test_size=val_size, shuffle=False)

        return train_data, val_data, test_data

class DataPreparation:
    def __init__(self):
        self.scaler = None
        
    def calculate_lags(self, df, max_lags=10, significance_level=0.05):
    # Define o número máximo de lags permitidos baseado no tamanho da série
        max_allowed_lags = min(max_lags, len(df["target"]) // 2 - 1)
        if max_allowed_lags <= 0:
            raise ValueError("Insufficient data points to calculate lags. Provide a longer series.")

        # Calcula a PACF
        partial_autocorr = pacf(df["target"], nlags=max_allowed_lags)
        n = len(df["target"])
        z_alpha = norm.ppf(1 - significance_level / 2)
        conf_interval = z_alpha / np.sqrt(n)
        
        # Identifica lags significativos
        self.lags = [i for i, coef in enumerate(partial_autocorr) if abs(coef) > conf_interval and i != 0]
        return self.lags

    def prepare_data(self, df_train, df_val, target_column, significative_lags):
        lags = significative_lags
        max_lags_validos = len(df_val) - 1  
        lags = [lag for lag in significative_lags if lag <= max_lags_validos]

        X_train, y_train = self.create_lags(df_train, target_column, lags)
        X_val, y_val = self.create_lags(df_val, target_column, lags)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        return X_train, y_train, X_val, y_val

    def create_lags(self, df, target_column, lags):
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        df.dropna(subset=[f'{target_column}_lag_{lag}' for lag in lags], inplace=True)
        lag_columns = [f'{target_column}_lag_{lag}' for lag in lags]
        X = df[lag_columns]
        y = df[target_column]

        return X, y

class PredictionModel:
    def __init__(self, param_grid=None):
        self.svr_model = None
        self.param_grid = param_grid or {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1e-3, 1e-2, 0.1, 1],
            'epsilon': [0.1, 0.2, 0.5],
            'kernel': ['linear', 'poly', 'rbf']  
        }

    def train_svr(self, X_train, y_train):
        grid_search = GridSearchCV(SVR(), self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.svr_model = grid_search.best_estimator_
        return self.svr_model

    def predict_svr(self, series, steps_ahead):
        data_preparation = DataPreparation()
        lags = data_preparation.calculate_lags(series, max_lags=10)
        X_train, y_train, X_val, _ = data_preparation.prepare_data(
            series, series, target_column='target', significative_lags=lags
        )
        self.train_svr(X_train, y_train)
        predictions = []
        last_known_data = X_val[-1] 
        for _ in range(steps_ahead):
            pred = self.svr_model.predict([last_known_data])[0]
            predictions.append(pred)
            last_known_data = np.roll(last_known_data, -1)  
            last_known_data[-1] = pred  

        return predictions, self.svr_model

    def predict_svr_online(self, model, y_true, series, steps_ahead):

        data_preparation = DataPreparation()
        lags = data_preparation.calculate_lags(series, max_lags=10)
        _, _, X_val, _ = data_preparation.prepare_data(series, series, 'target', lags)
        online_predictions = []
        last_known_data = X_val[-1]
        for i, true_val in enumerate(y_true):
            model.fit([last_known_data], [true_val])  
            pred = model.predict([last_known_data])[0]
            online_predictions.append(pred)

            last_known_data = np.roll(last_known_data, -1)
            last_known_data[-1] = pred

        return online_predictions

    






    
   





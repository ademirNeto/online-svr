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
        partial_autocorr = pacf(df["target"], nlags=max_lags)
        n = len(df["target"])
        z_alpha = norm.ppf(1 - significance_level / 2)
        conf_interval = z_alpha / np.sqrt(n)
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
            'kernel': ['linear', 'poli', 'rbf']  
        }

    def train_svr(self, X_train, y_train):
        grid_search = GridSearchCV(SVR(), self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.svr_model = grid_search.best_estimator_
        return self.svr_model
    
 
    def svr_online(self, X_train, y_train, X_val, y_val, serie_name, relevance_threshold=0.1, max_support_vectors=5000):
     
        self.svr_model = self.train_svr(X_train, y_train, serie_name=serie_name)
        y_pred_online = []
        X_buffer = X_train
        y_buffer = y_train

        for i in range(len(X_val)):
            X_new = X_val[i]
            y_new = y_val.iloc[i]
            y_pred_atual = self.svr_model.predict([X_new])[0]
            y_pred_online.append(y_pred_atual)
            erro = abs(y_new - y_pred_atual)
            if erro > relevance_threshold:
                X_buffer = np.vstack([X_buffer, X_new])
                y_buffer = np.append(y_buffer, y_new)
                self.svr_model.fit(X_buffer, y_buffer)
                if len(self.svr_model.support_vectors_) > max_support_vectors:
                    indices_relevantes = np.argsort(np.abs(self.svr_model.dual_coef_.flatten()))[::-1]
                    indices_mantidos = indices_relevantes[:max_support_vectors]
                    X_buffer = self.svr_model.support_vectors_[indices_mantidos]
                    y_buffer = y_buffer[indices_mantidos]
        return np.array(y_pred_online)


    






    
   





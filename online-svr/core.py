import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf
import numpy as np
from scipy.stats import norm
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
#import shap
from dieboldmariano import dm_test
from timeseriesmetrics import theil, wpocid
import pickle
import os
#import random
#from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")


class Serie_analisys:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_and_split(self, test_size=0.2, val_size=0.5):
 
        self.data = pd.read_csv(self.file_path, parse_dates=["date"])
        train_data, temp_data = train_test_split(self.data, test_size=test_size, shuffle=False)  
        val_data, test_data = train_test_split(temp_data, test_size=val_size, shuffle=False)
        
        '''
        print("Tamanho da série completa:", len(self.data))
        print("Tamanho do conjunto de treinamento:", len(train_data))
        print("Tamanho do conjunto de validação:", len(val_data))
        print("Tamanho do conjunto de teste:", len(test_data))
        #print(f"Perído da série: {self.data.index.min().strftime('%d-%m-%Y')} até {self.data.index.max().strftime('%d-%m-%Y')}")


        print("\nDados de Treinamento:")
        print(train_data.head())
        
        print("\nEstatísticas descritivas do conjunto de treinamento:")
        print(train_data.describe())
        '''
        
        return train_data, val_data, test_data

    def plot_splits(self, train_data, val_data, test_data):
        # Plotando as séries temporais
        val_plot = pd.concat([train_data.tail(1), val_data])
        test_plot = pd.concat([val_data.tail(1), test_data])

        plt.figure(figsize=(14, 6))

        # Plotando os conjuntos de treino, validação e teste
        plt.plot(train_data.index, train_data["target"], label='Treino', color='blue')
        plt.plot(val_plot.index, val_plot["target"], label='Validação', color='green')
        plt.plot(test_plot.index, test_plot["target"], label='Teste', color='red')

        # Configurações do gráfico
        plt.title('Série Temporal - Treinamento, Validação e Teste')
        plt.xlabel('Data')
        plt.ylabel("target")
        plt.legend()
        #plt.show()
        plt.savefig("../SVR/output/partioned_serie.png")

    def decompor_serie_boxcox(self, df, coluna, periodo=365, modelo='additive'):
  
        df[coluna] = df[coluna].apply(lambda x: x if x > 0 else 1)
        decomposicao_original = seasonal_decompose(df[coluna], model=modelo, period=periodo)
        plt.figure(figsize=(10, 8))
        decomposicao_original.plot()
        #plt.suptitle(f"Decomposição Original - {coluna}", fontsize=16)
        plt.savefig("SVR/output/box-cox.png")


        df[f'{coluna}_boxcox'], lam = boxcox(df[coluna])
        decomposicao_boxcox = seasonal_decompose(df[f'{coluna}_boxcox'], model=modelo, period=periodo)
        plt.figure(figsize=(10, 8))
        decomposicao_boxcox.plot()
        #plt.suptitle(f"Decomposição após Box-Cox - {coluna}", fontsize=16)
        plt.savefig("SVR/output/boxcox_differentied.png")
        print(f"Lambda utilizado na transformação de Box-Cox: {lam}")
        #return decomposicao_boxcox, lam


class DataPreparation:
    def __init__(self, param_grid=None):
        self.scaler = None
        if param_grid is None:
            self.param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1e-3, 1e-2, 0.1, 1],
                'epsilon': [0.1, 0.2, 0.5]
            }
        else:
            self.param_grid = param_grid

    def calculate_lags(self, df, max_lags=10, significance_level=0.05):
        partial_autocorr = pacf(df["target"], nlags=max_lags)
        n = len(df["target"])
        z_alpha = norm.ppf(1 - significance_level / 2)
        conf_interval = z_alpha / np.sqrt(n)
        self.lags = [i for i, coef in enumerate(partial_autocorr) if abs(coef) > conf_interval and i != 0]
        return self.lags

    def prepare_data(self, df_train, df_val, target_column, significative_lags):
        # Calcular lags
        lags = significative_lags
        max_lags_validos = len(df_val) - 1  # Número máximo de lags possíveis no conjunto de validação
        lags = [lag for lag in significative_lags if lag <= max_lags_validos]

        # Preparar dados de treino e validação
        X_train, y_train = self.create_lags(df_train, target_column, lags)
        X_val, y_val = self.create_lags(df_val, target_column, lags)

        # Normalizar os dados
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        return X_train, y_train, X_val, y_val

    def create_lags(self, df, target_column, lags):
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        # Remover linhas com NaN apenas para os lags especificados
        df.dropna(subset=[f'{target_column}_lag_{lag}' for lag in lags], inplace=True)
        # Selecionar as colunas de entrada (lags) e o alvo
        lag_columns = [f'{target_column}_lag_{lag}' for lag in lags]
        X = df[lag_columns]
        y = df[target_column]

        return X, y


    def plot_autocorrelations(self, df, target_column, max_lags):
        plt.figure(figsize=(12, 6))
        plot_acf(df[target_column], lags=max_lags)
        plt.title('Autocorrelação')
        plt.savefig("output/acf.png")

        plt.figure(figsize=(12, 6))
        plot_pacf(df[target_column], lags=max_lags)
        plt.title('Autocorrelação Parcial')
        plt.savefig("output/pacf.png")


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
    
    def get_or_train_model(self, X_train, y_train,serie_name):
        if os.path.exists(f'SVR/models/{serie_name}_svr_model.pkl'):
            with open(f'SVR/models/{serie_name}_svr_model.pkl', 'rb') as f:
                self.svr_model = pickle.load(f)
        else:
            grid_search = GridSearchCV(SVR(kernel='rbf'), self.param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            self.svr_model = grid_search.best_estimator_
            with open(f'SVR/models/{serie_name}_svr_model.pkl', 'wb') as f:
                pickle.dump(self.svr_model, f)
        
        return self.svr_model


    def svr_predict(self, X_train, y_train, X_val, serie_name):
        svr_model = self.get_or_train_model(X_train, y_train,serie_name=serie_name)
        y_pred = svr_model.predict(X_val)
       
        return y_pred
    '''
    def svr_online_predict(self, X_train, y_train, X_val, y_val, threshold=1.83):
        if self.train_svr is None:
            self.train_svr(X_train, y_train)

        svr_online_model = self.svr_model
        y_pred_online = []

        for i in range(len(X_val)):
            X_new = X_val[i]
            y_new = y_val.iloc[i]
            svr_online_model = self.atualizar_svr(svr_online_model, X_new, y_new, threshold=threshold)
            y_pred_atual = svr_online_model.predict([X_new])[0]
            y_pred_online.append(y_pred_atual)
        
        scores = cross_val_score(svr_online_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

        return np.array(y_pred_online), scores

    def atualizar_svr(self, svr_model, X_new, y_new, threshold=0.1):
        y_pred_new = svr_model.predict([X_new])[0]
        erro = y_new - y_pred_new

        if abs(erro) > threshold:
            vetores_suporte = svr_model.support_vectors_
            alphas = svr_model.dual_coef_.flatten()
            contribuicoes = self.calcular_contribuicao(vetores_suporte, alphas)
            idx_min_contribuicao = np.argmin(contribuicoes)

            if idx_min_contribuicao < len(vetores_suporte):
                vetores_suporte[idx_min_contribuicao] = X_new
                alphas[idx_min_contribuicao] = erro
            else:
                vetores_suporte = np.vstack([vetores_suporte, X_new])
                alphas = np.append(alphas, erro)

            svr_model.fit(vetores_suporte, y_new + alphas)

        return svr_model
    '''
    def random_walk_predict(self, y_val,):
        y_pred = y_val.shift(1).bfill()
        return y_pred

    def calculate_metrics(self, y_true, y_pred):
        y_true = y_true.copy()
        y_pred = y_pred.copy()
        mse = round(mean_squared_error(y_true, y_pred), 2)
        mape = round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2)
        pocid = round(wpocid(y_true=y_true, y_pred=y_pred), 2)
        u_theil = theil(y_pred=y_pred, y_true=y_true)
        return mse, mape, pocid, u_theil



    def get_results_dataframe(self, model_name, y_true, y_pred, exec_time, memory_usage):
        mse, mape, pocid, u_theil = self.calculate_metrics(y_true, y_pred)
        
        # Criação do DataFrame com os resultados
        results_df = pd.DataFrame({
            'Model': [model_name],
            'MSE': [mse],
            'MAPE (%)': [mape],
            'WPOCID': [pocid],
            'U de Theil': [u_theil],
            'Execution Time (s)': [exec_time],
            'Memory Usage (MB)': [memory_usage]
        })
        
        return results_df

        
    def diebold_mariano(self, y_true, y_pred_modelo1, y_pred_modelo2,nome_modelo1,nome_modelo2):

        estatistica_dm, p_valor = dm_test(y_true, y_pred_modelo1, y_pred_modelo2, one_sided=True)

        print(f"Comparação entre {nome_modelo1} e {nome_modelo2}")
        print(f"Estatística Diebold-Mariano: {estatistica_dm:.4f}")
        print(f"P-valor: {p_valor:.4f}")


    ##########
    
    
    def svr_online_update_safe(self, X_train, y_train, X_val, y_val, serie_name, relevance_threshold=0.1, max_support_vectors=5000):
        """
        Aprendizagem online para SVR com atualização segura e sem alterar diretamente a estrutura interna do modelo.

        Args:
            X_train (array-like): Dados de treinamento iniciais.
            y_train (array-like): Labels correspondentes ao X_train.
            X_val (array-like): Dados para previsão e atualização online.
            y_val (array-like): Labels correspondentes ao X_val.
            serie_name (str): Nome da série, usado para armazenar o modelo.
            relevance_threshold (float): Limite para considerar uma amostra relevante para atualização.
            max_support_vectors (int): Número máximo de vetores de suporte permitidos.

        Returns:
            np.array: Previsões incrementais realizadas pelo modelo atualizado.
        """
        # Treina o modelo inicial offline
        self.svr_model = self.get_or_train_model(X_train, y_train, serie_name=serie_name)
        y_pred_online = []

        # Inicializar buffer para dados acumulados
        X_buffer = X_train
        y_buffer = y_train

        for i in range(len(X_val)):
            X_new = X_val[i]
            y_new = y_val.iloc[i]

            # Prever o valor atual
            y_pred_atual = self.svr_model.predict([X_new])[0]
            y_pred_online.append(y_pred_atual)

            # Calcula o erro e avalia relevância
            erro = abs(y_new - y_pred_atual)
            if erro > relevance_threshold:
                # Adiciona nova evidência ao buffer
                X_buffer = np.vstack([X_buffer, X_new])
                y_buffer = np.append(y_buffer, y_new)

                # Reajustar o modelo usando os dados acumulados
                self.svr_model.fit(X_buffer, y_buffer)

                # Limitar o número de vetores de suporte se necessário
                if len(self.svr_model.support_vectors_) > max_support_vectors:
                    # Selecionar os vetores de suporte mais relevantes (com base no erro ou proximidade)
                    indices_relevantes = np.argsort(np.abs(self.svr_model.dual_coef_.flatten()))[::-1]
                    indices_mantidos = indices_relevantes[:max_support_vectors]
                    X_buffer = self.svr_model.support_vectors_[indices_mantidos]
                    y_buffer = y_buffer[indices_mantidos]

        return np.array(y_pred_online)


    






    
   





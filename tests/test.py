#from online-svr.core import PredictionModel

# Criar a instância do modelo
model = PredictionModel()

# Sua série temporal como DataFrame
import pandas as pd
series = pd.DataFrame({'target': [1.1, 2.3, 3.4, 4.5, 5.6]})

# Prever 5 passos à frente
predictions, trained_model = model.predict_svr(series, steps_ahead=5)

print("Previsões:", predictions)

# Criar um modelo inicial e treinar
predictions, trained_model = model.predict_svr(series, steps_ahead=5)

# Novos dados de verdade para atualizar o modelo online
y_true = pd.Series([6.7, 7.8, 8.9])

# Atualizar o modelo e prever os próximos 3 passos
online_predictions = model.predict_svr_online(trained_model, y_true, series, steps_ahead=3)

print("Previsões após atualização online:", online_predictions)

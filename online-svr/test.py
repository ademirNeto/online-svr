import sys
sys.path.append(r"C:\Users\neton\AppData\Local\Programs\Python\Python311\Lib\site-packages")
import online_svr
print("Pacote importado com sucesso!")


"""
from core import PredictionModel
import pandas as pd

model = PredictionModel()

series = pd.DataFrame({
    'target': [
        1.1, 2.3, 3.6, 4.8, 6.1, 7.5, 8.9, 10.4, 12.0, 13.7,
        15.3, 16.9, 18.5, 20.1, 21.8, 23.5, 25.3, 27.1, 28.9, 30.7
    ]
})

predictions, trained_model = model.predict_svr(series, steps_ahead=5)
print("Previsões offline:", predictions)

y_true = pd.Series([31.5, 33.2, 34.8])

online_predictions = model.predict_svr_online(trained_model, y_true, series, steps_ahead=3)
print("Previsões após atualização online:", online_predictions)
"""
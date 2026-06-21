import numpy as np
from online_svr import LagTransformer, SVRForecaster

np.random.seed(42)
y = np.array([i * 1.5 + 3 * np.sin(i * 0.4) for i in range(60)])

# Lag features
transformer = LagTransformer(max_lags=10, significance_level=0.05)
transformer.fit(y)
X = transformer.transform(y)
y_aligned = y[max(transformer.lags_):]

print("Significant lags:", transformer.lags_)

# --- Modo padrão: treino/teste interno, previsao multi-step ---
model = SVRForecaster(cv=3, test_size=0.2)
model.fit(X, y_aligned)
print("Tamanho treino:", len(X) - len(model.X_test_))
print("Tamanho teste: ", len(model.X_test_))

preds = model.predict(model.X_test_[-1:], steps=5)
print("Multi-step (5 passos):", np.round(preds, 2))

# --- Modo one_step_ahead: treina em tudo, preve so o proximo ---
model_osa = SVRForecaster(cv=3, one_step_ahead=True)
model_osa.fit(X, y_aligned)

next_step = model_osa.predict(X[-1:])
print("One-step-ahead:       ", np.round(next_step, 2))

# --- Online update ---
model.partial_fit(model.X_test_[-1:], [model.y_test_[-1]])
updated = model.predict(model.X_test_[-1:], steps=1)
print("Apos partial_fit:     ", np.round(updated, 2))

import numpy as np
import pandas as pd

def criar_features(dados):
    dados_ml = dados.copy()

    dados_ml["lag_1"] = dados_ml["consumo_zona_1"].shift(1)
    dados_ml["lag_6"] = dados_ml["consumo_zona_1"].shift(6)
    dados_ml["lag_144"] = dados_ml["consumo_zona_1"].shift(144)
    dados_ml["lag_1008"] = dados_ml["consumo_zona_1"].shift(1008)

    dados_ml["rolling_mean_6"] = dados_ml["consumo_zona_1"].shift(1).rolling(6).mean()
    dados_ml["rolling_mean_144"] = dados_ml["consumo_zona_1"].shift(1).rolling(144).mean()

    dados_ml["dia_semana"] = dados_ml.index.dayofweek
    dados_ml["hora"] = dados_ml.index.hour

    dados_ml["hora_sin"] = np.sin(2 * np.pi * dados_ml["hora"] / 24)
    dados_ml["hora_cos"] = np.cos(2 * np.pi * dados_ml["hora"] / 24)

    dados_ml.drop(columns="hora", inplace=True)

    return dados_ml.dropna()
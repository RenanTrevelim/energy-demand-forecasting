import numpy as np
import pandas as pd


TARGET = "consumo_zona_1"

VARIAVEIS_CLIMA = [
    "temperatura",
    "umidade",
    "velocidade_vento",
    "fluxo_difuso_geral",
    "fluxo_difuso"
]

def forecast_mensal(
    modelo,
    dados_ml: pd.DataFrame,
    horizonte: int = 4320
) -> pd.DataFrame:

    datas_futuras = pd.date_range(
        start=dados_ml.index.max() + pd.Timedelta(minutes=10),
        periods=horizonte,
        freq="10min"
    )

    clima_futuro = (
        dados_ml[VARIAVEIS_CLIMA]
        .iloc[-horizonte:]
        .copy()
    )

    clima_futuro.index = datas_futuras

    historico = dados_ml[TARGET].copy()

    colunas_modelo = list(
        modelo.feature_names_in_
    )

    previsoes = []

    for data_hora, clima in clima_futuro.iterrows():

        X_futuro = pd.DataFrame({
            "temperatura": [clima["temperatura"]],
            "umidade": [clima["umidade"]],
            "velocidade_vento": [clima["velocidade_vento"]],
            "fluxo_difuso_geral": [clima["fluxo_difuso_geral"]],
            "fluxo_difuso": [clima["fluxo_difuso"]],

            "lag_1": [historico.iloc[-1]],
            "lag_6": [historico.iloc[-6]],
            "lag_144": [historico.iloc[-144]],
            "lag_1008": [historico.iloc[-1008]],

            "rolling_mean_6": [
                historico.iloc[-6:].mean()
            ],

            "rolling_mean_144": [
                historico.iloc[-144:].mean()
            ],

            "dia_semana": [
                data_hora.dayofweek
            ],

            "hora_sin": [
                np.sin(
                    2 * np.pi
                    * data_hora.hour / 24
                )
            ],

            "hora_cos": [
                np.cos(
                    2 * np.pi
                    * data_hora.hour / 24
                )
            ]
        }, index=[data_hora])

        X_futuro = X_futuro[
            colunas_modelo
        ]

        previsao = modelo.predict(
            X_futuro
        )[0]

        previsoes.append(previsao)

        historico.loc[data_hora] = previsao

    return pd.DataFrame(
        {
            "consumo_previsto": previsoes
        },
        index=datas_futuras
    )
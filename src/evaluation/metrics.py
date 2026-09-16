import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


def calcular_metricas(
    y_real,
    y_previsto
):

    mae = mean_absolute_error(
        y_real,
        y_previsto
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            y_previsto
        )
    )

    mape = mean_absolute_percentage_error(
        y_real,
        y_previsto
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
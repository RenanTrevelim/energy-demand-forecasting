import matplotlib.pyplot as plt


def plot_real_vs_previsto(
    y_real,
    y_previsto
):

    plt.figure(figsize=(14, 6))

    plt.plot(
        y_real.index,
        y_real,
        label="Real",
        color="steelblue"
    )

    plt.plot(
        y_real.index,
        y_previsto,
        label="Previsto",
        color="darkorange",
        linestyle="--"
    )

    plt.title(
        "Consumo Real vs Previsto"
    )

    plt.xlabel("Data")
    plt.ylabel("Consumo de Energia")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()

def plot_forecast(
    historico,
    forecast
):

    historico_horario = (
        historico
        .resample("h")
        .mean()
    )

    forecast_horario = (
        forecast["consumo_previsto"]
        .resample("h")
        .mean()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        historico_horario.index,
        historico_horario,
        label="Histórico",
        color="steelblue"
    )

    plt.plot(
        forecast_horario.index,
        forecast_horario,
        label="Forecast",
        color="darkorange",
        linestyle="--"
    )

    plt.axvline(
        historico.index.max(),
        color="black",
        linestyle="--",
        alpha=0.7
    )

    plt.title(
        "Forecast do Consumo de Energia"
    )

    plt.xlabel("Data")
    plt.ylabel("Consumo Médio Horário")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()
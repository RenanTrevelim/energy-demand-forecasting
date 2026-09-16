from pathlib import Path
from textwrap import dedent

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CAMINHOS
# ==================================================
ROOT = Path(__file__).resolve().parent

CAMINHO_DADOS = ROOT / "data" / "dados_tratados.csv"
CAMINHO_MODELO = ROOT / "models" / "pipeline_final.pkl"


# ==================================================
# CONFIGURAÇÕES DO PROJETO
# ==================================================
TARGET = "consumo_zona_1"

HORIZONTE = 4320  # 30 dias em intervalos de 10 minutos
FREQUENCIA = "10min"

VARIAVEIS_CLIMA = [
    "temperatura",
    "umidade",
    "velocidade_vento",
    "fluxo_difuso_geral",
    "fluxo_difuso",
]


# ==================================================
# HTML
# ==================================================
def renderizar_html(conteudo: str) -> None:
    st.html(dedent(conteudo).strip())


# ==================================================
# ESTILO VISUAL
# ==================================================
renderizar_html(
    """
    <style>

        .stApp,
        [data-testid="stAppViewContainer"] {
            background-color: #F4F7FB;
            color: #0F172A;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0F172A 0%,
                #164E63 100%
            );
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #FFFFFF !important;
        }

        /* Hero */
        .hero {
            padding: 2.5rem 2.7rem;
            border-radius: 24px;

            background: linear-gradient(
                135deg,
                #0F172A 0%,
                #0F766E 55%,
                #06B6D4 100%
            );

            margin-bottom: 1.7rem;

            box-shadow:
                0 18px 45px
                rgba(15, 118, 110, 0.18);
        }

        .hero h1 {
            margin: 0;

            color: #FFFFFF !important;

            font-size: 2.7rem;
            line-height: 1.15;
        }

        .hero p {
            max-width: 980px;

            margin-top: 1rem;
            margin-bottom: 0;

            color: #CCFBF1 !important;

            font-size: 1.05rem;
            line-height: 1.7;
        }

        .badge {
            display: inline-block;

            padding: 0.4rem 0.9rem;
            margin-bottom: 1rem;

            border-radius: 999px;

            background-color:
                rgba(255, 255, 255, 0.18);

            color: #FFFFFF !important;

            font-size: 0.8rem;
            font-weight: 800;
        }

        /* Títulos */
        .section-title {
            margin-top: 1.9rem;
            margin-bottom: 0.45rem;

            color: #0F172A !important;

            font-size: 1.45rem;
            font-weight: 800;
        }

        .section-subtitle {
            margin-bottom: 1.2rem;

            color: #64748B !important;

            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* Cards */
        .info-card {
            min-height: 185px;

            padding: 1.55rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background: #FFFFFF;

            box-shadow:
                0 8px 25px
                rgba(15, 23, 42, 0.05);
        }

        .info-card h3 {
            margin-top: 0;

            color: #0F766E !important;
        }

        .info-card p {
            color: #475569 !important;
            line-height: 1.65;
        }

        /* KPI */
        .kpi-card {
            min-height: 135px;

            padding: 1.25rem 1.35rem;

            border: 1px solid #E2E8F0;
            border-radius: 18px;

            background:
                linear-gradient(
                    145deg,
                    #FFFFFF,
                    #F8FAFC
                );

            box-shadow:
                0 8px 22px
                rgba(15, 23, 42, 0.05);
        }

        .kpi-label {
            color: #64748B !important;

            font-size: 0.76rem;
            font-weight: 800;

            letter-spacing: 0.04rem;
            text-transform: uppercase;
        }

        .kpi-value {
            margin-top: 0.45rem;

            color: #0F172A !important;

            font-size: 1.75rem;
            font-weight: 800;
        }

        .kpi-detail {
            margin-top: 0.35rem;

            color: #64748B !important;

            font-size: 0.8rem;
        }

        /* Insight */
        .insight-card {
            padding: 1.5rem;

            border: 1px solid #99F6E4;
            border-radius: 18px;

            background:
                linear-gradient(
                    145deg,
                    #F0FDFA,
                    #FFFFFF
                );

            box-shadow:
                0 8px 24px
                rgba(15, 118, 110, 0.06);
        }

        .insight-card h3 {
            margin-top: 0;

            color: #0F766E !important;
        }

        .insight-card p {
            color: #475569 !important;

            line-height: 1.7;
        }

        .highlight {
            color: #0F766E !important;
            font-weight: 800;
        }

        /* Nota */
        .model-note {
            padding: 1.1rem 1.3rem;

            border-left: 5px solid #0F766E;
            border-radius: 14px;

            background-color: #ECFDF5;

            color: #065F46 !important;

            line-height: 1.65;
        }

        /* DataFrame */
        [data-testid="stDataFrame"] {
            overflow: hidden;

            border: 1px solid #DCE4F0;
            border-radius: 16px;

            box-shadow:
                0 8px 22px
                rgba(15, 23, 42, 0.05);
        }

        /* Download */
        [data-testid="stDownloadButton"] button {
            width: 100%;

            border: none;
            border-radius: 11px;

            background:
                linear-gradient(
                    90deg,
                    #0F766E,
                    #06B6D4
                );

            color: #FFFFFF;

            font-weight: 700;
        }

        /* Footer */
        .footer {
            margin-top: 3rem;

            padding-top: 1.5rem;

            border-top:
                1px solid #DCE4F0;

            color: #64748B !important;

            font-size: 0.88rem;

            text-align: center;
        }

    </style>
    """
)


# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================
def formatar_numero(valor: float, casas: int = 0) -> str:

    if pd.isna(valor):
        return "-"

    texto = f"{valor:,.{casas}f}"

    return (
        texto
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def converter_csv(dados: pd.DataFrame) -> bytes:

    return dados.to_csv(
        index=True,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


# ==================================================
# CARREGAMENTO DOS DADOS
# ==================================================
@st.cache_data(show_spinner=False)
def carregar_dados() -> pd.DataFrame:

    if not CAMINHO_DADOS.exists():
        raise FileNotFoundError(
            f"Base não encontrada: {CAMINHO_DADOS}"
        )

    dados = pd.read_csv(
        CAMINHO_DADOS
    )

    if "data_hora" not in dados.columns:
        raise ValueError(
            "A coluna 'data_hora' não foi encontrada."
        )

    dados["data_hora"] = pd.to_datetime(
        dados["data_hora"]
    )

    dados = (
        dados
        .set_index("data_hora")
        .sort_index()
    )

    return dados


# ==================================================
# FEATURE ENGINEERING
# ==================================================
@st.cache_data(show_spinner=False)
def preparar_dados_ml(
    dados: pd.DataFrame,
) -> pd.DataFrame:

    dados_ml = dados.copy()

    dados_ml["lag_1"] = (
        dados_ml[TARGET]
        .shift(1)
    )

    dados_ml["lag_6"] = (
        dados_ml[TARGET]
        .shift(6)
    )

    dados_ml["lag_144"] = (
        dados_ml[TARGET]
        .shift(144)
    )

    dados_ml["lag_1008"] = (
        dados_ml[TARGET]
        .shift(1008)
    )

    dados_ml["rolling_mean_6"] = (
        dados_ml[TARGET]
        .shift(1)
        .rolling(6)
        .mean()
    )

    dados_ml["rolling_mean_144"] = (
        dados_ml[TARGET]
        .shift(1)
        .rolling(144)
        .mean()
    )

    dados_ml["dia_da_semana"] = (
        dados_ml.index.dayofweek
    )

    dados_ml["hora_sin"] = np.sin(
        2
        * np.pi
        * dados_ml.index.hour
        / 24
    )

    dados_ml["hora_cos"] = np.cos(
        2
        * np.pi
        * dados_ml.index.hour
        / 24
    )

    return dados_ml.dropna()


# ==================================================
# CARREGAMENTO DO PIPELINE
# ==================================================
@st.cache_resource(show_spinner=False)
def carregar_modelo():

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    return joblib.load(
        CAMINHO_MODELO
    )


# ==================================================
# FORECAST RECURSIVO
# ==================================================
def gerar_forecast_mensal(
    modelo,
    dados_ml: pd.DataFrame,
) -> pd.DataFrame:

    datas_futuras = pd.date_range(
        start=(
            dados_ml.index.max()
            + pd.Timedelta(minutes=10)
        ),
        periods=HORIZONTE,
        freq=FREQUENCIA,
    )

    clima_futuro = (
        dados_ml[VARIAVEIS_CLIMA]
        .iloc[-HORIZONTE:]
        .copy()
    )

    clima_futuro.index = (
        datas_futuras
    )

    historico = (
        dados_ml[TARGET]
        .copy()
    )

    if hasattr(
        modelo,
        "feature_names_in_",
    ):
        colunas_modelo = list(
            modelo.feature_names_in_
        )

    else:
        colunas_modelo = (
            dados_ml
            .drop(columns=TARGET)
            .columns
            .tolist()
        )

    previsoes = []

    for data_hora, clima in clima_futuro.iterrows():

        X_futuro = pd.DataFrame(
            {
                "temperatura": [
                    clima["temperatura"]
                ],

                "umidade": [
                    clima["umidade"]
                ],

                "velocidade_vento": [
                    clima["velocidade_vento"]
                ],

                "fluxo_difuso_geral": [
                    clima["fluxo_difuso_geral"]
                ],

                "fluxo_difuso": [
                    clima["fluxo_difuso"]
                ],

                "lag_1": [
                    historico.iloc[-1]
                ],

                "lag_6": [
                    historico.iloc[-6]
                ],

                "lag_144": [
                    historico.iloc[-144]
                ],

                "lag_1008": [
                    historico.iloc[-1008]
                ],

                "rolling_mean_6": [
                    historico
                    .iloc[-6:]
                    .mean()
                ],

                "rolling_mean_144": [
                    historico
                    .iloc[-144:]
                    .mean()
                ],

                "dia_da_semana": [
                    data_hora.dayofweek
                ],

                "hora_sin": [
                    np.sin(
                        2
                        * np.pi
                        * data_hora.hour
                        / 24
                    )
                ],

                "hora_cos": [
                    np.cos(
                        2
                        * np.pi
                        * data_hora.hour
                        / 24
                    )
                ],
            },
            index=[data_hora],
        )

        X_futuro = X_futuro[
            colunas_modelo
        ]

        previsao = modelo.predict(
            X_futuro
        )[0]

        previsoes.append(
            previsao
        )

        historico.loc[
            data_hora
        ] = previsao

    return pd.DataFrame(
        {
            "consumo_previsto":
                previsoes
        },
        index=datas_futuras,
    )


# ==================================================
# EXECUÇÃO DO FORECAST
# ==================================================
@st.cache_data(show_spinner=False)
def executar_forecast():

    dados = carregar_dados()

    dados_ml = preparar_dados_ml(
        dados
    )

    modelo = carregar_modelo()

    forecast = gerar_forecast_mensal(
        modelo,
        dados_ml,
    )

    return dados_ml, forecast


# ==================================================
# INDICADORES
# ==================================================
def calcular_indicadores(
    dados_ml: pd.DataFrame,
    forecast: pd.DataFrame,
) -> dict:

    ultimo_mes = (
        dados_ml[TARGET]
        .iloc[-HORIZONTE:]
    )

    media_anterior = (
        ultimo_mes.mean()
    )

    media_prevista = (
        forecast[
            "consumo_previsto"
        ].mean()
    )

    variacao_pct = (
        (
            media_prevista
            - media_anterior
        )
        / media_anterior
        * 100
    )

    data_max = (
        forecast[
            "consumo_previsto"
        ].idxmax()
    )

    valor_max = (
        forecast.loc[
            data_max,
            "consumo_previsto",
        ]
    )

    data_min = (
        forecast[
            "consumo_previsto"
        ].idxmin()
    )

    valor_min = (
        forecast.loc[
            data_min,
            "consumo_previsto",
        ]
    )

    return {
        "media_anterior":
            media_anterior,

        "media_prevista":
            media_prevista,

        "variacao_pct":
            variacao_pct,

        "data_max":
            data_max,

        "valor_max":
            valor_max,

        "data_min":
            data_min,

        "valor_min":
            valor_min,
    }


# ==================================================
# GRÁFICO PRINCIPAL
# ==================================================
def criar_grafico_forecast(
    dados_ml: pd.DataFrame,
    forecast: pd.DataFrame,
):

    historico = (
        dados_ml[TARGET]
        .iloc[-HORIZONTE:]
        .resample("D")
        .mean()
    )

    previsao = (
        forecast["consumo_previsto"]
        .resample("D")
        .mean()
    )

    inicio_forecast = previsao.index.min()
    fim_forecast = previsao.index.max()

    data_pico = previsao.idxmax()
    valor_pico = previsao.max()

    data_min = previsao.idxmin()
    valor_min = previsao.min()

    fig, ax = plt.subplots(figsize=(16, 7))

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Área sombreada do forecast
    ax.axvspan(
        inicio_forecast,
        fim_forecast,
        color="#FEF3C7",
        alpha=0.18,
        label="_nolegend_"
    )

    # Histórico
    ax.plot(
        historico.index,
        historico.values,
        color="#2563EB",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="Últimos 30 dias"
    )

    # Forecast
    ax.plot(
        previsao.index,
        previsao.values,
        color="#F59E0B",
        linewidth=2.4,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Próximos 30 dias"
    )

    # Linha divisória
    ax.axvline(
        historico.index.max(),
        color="#64748B",
        linestyle="--",
        linewidth=1.3,
        alpha=0.9
    )

    # Pico previsto
    ax.scatter(
        data_pico,
        valor_pico,
        color="#D97706",
        s=70,
        zorder=5
    )

    ax.annotate(
        f"Pico: {valor_pico:,.0f}".replace(",", "."),
        xy=(data_pico, valor_pico),
        xytext=(15, 18),
        textcoords="offset points",
        fontsize=10,
        color="#92400E",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="#D97706",
            lw=1.2
        )
    )

    # Menor valor previsto
    ax.scatter(
        data_min,
        valor_min,
        color="#B45309",
        s=70,
        zorder=5
    )

    ax.annotate(
        f"Mínimo: {valor_min:,.0f}".replace(",", "."),
        xy=(data_min, valor_min),
        xytext=(15, -25),
        textcoords="offset points",
        fontsize=10,
        color="#92400E",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="#B45309",
            lw=1.2
        )
    )

    # Texto auxiliar
    ax.text(
        previsao.index[2],
        ax.get_ylim()[1] * 0.995,
        "Período previsto",
        fontsize=10,
        color="#92400E",
        fontweight="bold",
        va="top"
    )

    ax.set_title(
        "Histórico e Forecast de Consumo",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        pad=18
    )

    ax.set_xlabel(
        "Data",
        fontsize=11,
        color="#475569"
    )

    ax.set_ylabel(
        "Consumo Médio Diário",
        fontsize=11,
        color="#475569"
    )

    # Formatação do eixo X
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    plt.xticks(rotation=0)

    # Grid suave
    ax.grid(
        alpha=0.12,
        linestyle="--"
    )

    # Remove bordas
    ax.spines[
        ["top", "right", "left", "bottom"]
    ].set_visible(False)

    # Legenda
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=10
    )

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO DIÁRIO
# ==================================================
def criar_grafico_diario(
    forecast: pd.DataFrame,
):

    consumo_diario = (
        forecast[
            "consumo_previsto"
        ]
        .resample("D")
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(16, 5.5)
    )

    fig.patch.set_facecolor(
        "#FFFFFF"
    )

    ax.plot(
        consumo_diario.index,
        consumo_diario,
        color="#0F766E",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )

    ax.fill_between(
        consumo_diario.index,
        consumo_diario.values,
        alpha=0.08,
        color="#14B8A6",
    )

    ax.set_title(
        "Consumo Médio Previsto por Dia",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        pad=18,
    )

    ax.set_xlabel(
        "Data"
    )

    ax.set_ylabel(
        "Consumo Médio"
    )

    ax.grid(
        alpha=0.15
    )

    ax.spines[
        [
            "top",
            "right",
            "left",
            "bottom",
        ]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.markdown(
        "## ⚡ Energy Forecast"
    )

    st.caption(
        "Energy Demand Intelligence"
    )

    st.divider()

    pagina = st.radio(
        "Navegação",
        [
            "Visão geral",
            "Forecast mensal",
            "Sobre o projeto",
        ],
    )

    st.divider()

    st.markdown(
        "### Tecnologia"
    )

    st.markdown(
        """
**Machine Learning**

XGBoost + Optuna

**Time Series**

Lags + Rolling Features

**Produto**

Forecast de Demanda
        """
    )

    st.divider()

    st.caption(
        "Previsão e análise operacional "
        "do consumo energético."
    )


# ==================================================
# VISÃO GERAL
# ==================================================
if pagina == "Visão geral":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Machine Learning • Energy Intelligence
            </span>

            <h1>
                Energy Demand Forecasting
            </h1>

            <p>
                Aplicação de Machine Learning para análise
                e previsão da demanda energética, utilizando
                XGBoost e informações temporais para transformar
                dados históricos em projeções de consumo.
            </p>

        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            Da previsão à decisão operacional
        </div>

        <div class="section-subtitle">
            O modelo é utilizado para antecipar o comportamento
            da demanda e apoiar análises sobre consumo,
            períodos críticos e planejamento energético.
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        renderizar_html(
            """
            <div class="info-card">

                <h3>📈 Antecipar demanda</h3>

                <p>
                    Projeta o comportamento esperado
                    do consumo energético ao longo
                    dos próximos 30 dias.
                </p>

            </div>
            """
        )

    with col2:

        renderizar_html(
            """
            <div class="info-card">

                <h3>⚡ Identificar picos</h3>

                <p>
                    Localiza períodos de maior e menor
                    demanda prevista para facilitar
                    o acompanhamento operacional.
                </p>

            </div>
            """
        )

    with col3:

        renderizar_html(
            """
            <div class="info-card">

                <h3>🎯 Apoiar planejamento</h3>

                <p>
                    Compara o próximo ciclo previsto
                    com o comportamento recente para
                    contextualizar a evolução da demanda.
                </p>

            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Proposta de valor
        </div>
        """
    )

    renderizar_html(
        """
        <div class="model-note">

            O projeto transforma um modelo preditivo
            em uma aplicação analítica capaz de apresentar
            tendências futuras, períodos de maior demanda
            e indicadores resumidos de consumo.

        </div>
        """
    )


# ==================================================
# FORECAST MENSAL
# ==================================================
elif pagina == "Forecast mensal":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                30-Day Energy Demand Forecast
            </span>

            <h1>
                Forecast Mensal de Demanda
            </h1>

            <p>
                Simulação do consumo energético para os
                próximos 30 dias utilizando XGBoost,
                features temporais e previsão recursiva.
            </p>

        </div>
        """
    )

    try:

        with st.spinner(
            "Gerando forecast dos próximos 30 dias..."
        ):

            dados_ml, forecast = (
                executar_forecast()
            )

        indicadores = (
            calcular_indicadores(
                dados_ml,
                forecast,
            )
        )

        renderizar_html(
            """
            <div class="section-title">
                Visão executiva da previsão
            </div>

            <div class="section-subtitle">
                Principais indicadores projetados
                para o próximo ciclo de 30 dias.
            </div>
            """
        )

        kpi1, kpi2, kpi3, kpi4 = (
            st.columns(4)
        )

        with kpi1:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Consumo médio previsto
                    </div>

                    <div class="kpi-value">
                        {formatar_numero(
                            indicadores["media_prevista"]
                        )}
                    </div>

                    <div class="kpi-detail">
                        média dos próximos 30 dias
                    </div>

                </div>
                """
            )

        with kpi2:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Pico previsto
                    </div>

                    <div class="kpi-value">
                        {formatar_numero(
                            indicadores["valor_max"]
                        )}
                    </div>

                    <div class="kpi-detail">
                        {indicadores["data_max"].strftime(
                            "%d/%m %H:%M"
                        )}
                    </div>

                </div>
                """
            )

        with kpi3:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Menor demanda
                    </div>

                    <div class="kpi-value">
                        {formatar_numero(
                            indicadores["valor_min"]
                        )}
                    </div>

                    <div class="kpi-detail">
                        {indicadores["data_min"].strftime(
                            "%d/%m %H:%M"
                        )}
                    </div>

                </div>
                """
            )

        with kpi4:

            cor_variacao = (
                "#DC2626"
                if indicadores[
                    "variacao_pct"
                ] > 0
                else "#0F766E"
            )

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Variação mensal
                    </div>

                    <div
                        class="kpi-value"
                        style="color:
                        {cor_variacao}
                        !important;"
                    >
                        {indicadores["variacao_pct"]:+.2f}%
                    </div>

                    <div class="kpi-detail">
                        vs. últimos 30 dias
                    </div>

                </div>
                """
            )

        # ==================================================
        # GRÁFICO PRINCIPAL
        # ==================================================
        renderizar_html(
            """
            <div class="section-title">
                Evolução da demanda energética
            </div>

            <div class="section-subtitle">
                Comparação entre o último mês observado
                e o comportamento projetado para os
                próximos 30 dias.
            </div>
            """
        )

        grafico = criar_grafico_forecast(
            dados_ml,
            forecast,
        )

        st.pyplot(
            grafico,
            use_container_width=True,
        )

        plt.close(
            grafico
        )

        # ==================================================
        # LEITURA OPERACIONAL
        # ==================================================
        tendencia = (
            "aumento"
            if indicadores[
                "variacao_pct"
            ] > 0
            else "redução"
        )

        renderizar_html(
            f"""
            <div class="insight-card">

                <h3>
                    💡 Leitura operacional
                </h3>

                <p>
                    O modelo projeta consumo médio de
                    <span class="highlight">
                        {formatar_numero(
                            indicadores["media_prevista"]
                        )}
                    </span>
                    para os próximos 30 dias.
                </p>

                <p>
                    Em relação ao último mês observado,
                    a projeção representa
                    <span class="highlight">
                        {tendencia} de
                        {abs(
                            indicadores["variacao_pct"]
                        ):.2f}%
                    </span>.
                </p>

                <p>
                    O maior consumo previsto ocorre em
                    <span class="highlight">
                        {indicadores["data_max"].strftime(
                            "%d/%m/%Y às %H:%M"
                        )}
                    </span>,
                    atingindo aproximadamente
                    <span class="highlight">
                        {formatar_numero(
                            indicadores["valor_max"]
                        )}
                    </span>.
                </p>

            </div>
            """
        )

        # ==================================================
        # EVOLUÇÃO DIÁRIA
        # ==================================================
        renderizar_html(
            """
            <div class="section-title">
                Evolução diária prevista
            </div>

            <div class="section-subtitle">
                Agregação diária do forecast para facilitar
                a análise do comportamento da demanda
                ao longo do mês projetado.
            </div>
            """
        )

        grafico_diario = (
            criar_grafico_diario(
                forecast
            )
        )

        st.pyplot(
            grafico_diario,
            use_container_width=True,
        )

        plt.close(
            grafico_diario
        )

        # ==================================================
        # RESUMO DIÁRIO
        # ==================================================
        resumo_diario = (
            forecast[
                "consumo_previsto"
            ]
            .resample("D")
            .agg(
                consumo_medio="mean",
                consumo_maximo="max",
                consumo_minimo="min",
            )
            .reset_index()
        )

        renderizar_html(
            """
            <div class="section-title">
                Resumo diário da previsão
            </div>

            <div class="section-subtitle">
                Indicadores consolidados para cada
                dia do horizonte de forecast.
            </div>
            """
        )

        st.dataframe(
            resumo_diario,
            use_container_width=True,
            hide_index=True,
            column_config={
                "data_hora":
                    st.column_config.DateColumn(
                        "Data",
                        format="DD/MM/YYYY",
                    ),

                "consumo_medio":
                    st.column_config.NumberColumn(
                        "Consumo médio",
                        format="%.2f",
                    ),

                "consumo_maximo":
                    st.column_config.NumberColumn(
                        "Consumo máximo",
                        format="%.2f",
                    ),

                "consumo_minimo":
                    st.column_config.NumberColumn(
                        "Consumo mínimo",
                        format="%.2f",
                    ),
            },
        )

        # ==================================================
        # DOWNLOAD
        # ==================================================
        st.download_button(
            "⬇️ Exportar forecast mensal",
            data=converter_csv(
                forecast
            ),
            file_name=(
                "forecast_consumo_30_dias.csv"
            ),
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

        renderizar_html(
            """
            <div class="model-note">

                <strong>Importante:</strong>

                o forecast mensal é uma simulação recursiva.
                As variáveis meteorológicas futuras são
                representadas pelo comportamento observado
                nos últimos 30 dias.

                Em um ambiente produtivo, essas informações
                deveriam ser substituídas por previsões
                meteorológicas futuras.

            </div>
            """
        )

    except Exception as erro:

        st.error(
            "Não foi possível executar "
            f"o forecast: {erro}"
        )


# ==================================================
# SOBRE O PROJETO
# ==================================================
else:

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                End-to-End Machine Learning Project
            </span>

            <h1>
                Sobre o Energy Demand Forecasting
            </h1>

            <p>
                Projeto de portfólio desenvolvido para
                demonstrar análise temporal, Machine Learning,
                avaliação preditiva e transformação do modelo
                em uma aplicação analítica interativa.
            </p>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🤖 Machine Learning
                </h3>

                <p>
                    Feature Engineering temporal,
                    TimeSeriesSplit, XGBoost,
                    otimização com Optuna,
                    análise de resíduos e SHAP.
                </p>

            </div>
            """
        )

    with col2:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    ⚡ Produto de dados
                </h3>

                <p>
                    O modelo treinado é integrado
                    a uma aplicação Streamlit para
                    transformar previsões em indicadores
                    de demanda e informações operacionais.
                </p>

            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Arquitetura da solução
        </div>
        """
    )

    st.code(
        """
dados_tratados.csv
        ↓
Feature Engineering Temporal
        ↓
Lags + Rolling Features
        ↓
pipeline_final.pkl
        ↓
SimpleImputer
        ↓
XGBoost
        ↓
Forecast Recursivo
        ↓
30 dias / 4.320 previsões
        ↓
Indicadores Operacionais
        ↓
Streamlit Dashboard
        """,
        language="text",
    )

    renderizar_html(
        """
        <div class="section-title">
            Pipeline do modelo
        </div>
        """
    )

    st.code(
        """
pipeline_final.pkl

Pipeline(
    SimpleImputer(strategy="median")
        ↓
    XGBRegressor(...)
)
        """,
        language="text",
    )

    renderizar_html(
        """
        <div class="model-note">

            O pipeline persistido contém o pré-processamento
            e o modelo XGBoost final.

            Dessa forma, a aplicação carrega um único artefato
            para realizar as previsões, garantindo consistência
            entre treinamento e inferência.

        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            Tecnologias
        </div>
        """
    )

    st.code(
        """Python
Pandas
NumPy
Scikit-learn
XGBoost
Optuna
SHAP
Matplotlib
Streamlit
Joblib""",
        language="text",
    )


# ==================================================
# RODAPÉ
# ==================================================
renderizar_html(
    """
    <div class="footer">

        Energy Demand Forecasting •
        Machine Learning + Time Series Intelligence

    </div>
    """
)
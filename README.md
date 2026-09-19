# Energy Demand Forecasting — Time Series, Machine Learning & Ensemble Models

Projeto de Ciência de Dados para análise e previsão de demanda energética utilizando técnicas de séries temporais, Feature Engineering, Machine Learning e Ensemble Learning.

O objetivo principal é construir uma solução completa de previsão de consumo energético, desde a validação e análise temporal dos dados até a comparação de modelos, otimização de hiperparâmetros, interpretabilidade com SHAP, criação de pipeline e disponibilização das previsões em uma aplicação Streamlit.


---

## Objetivo do Projeto

Este projeto busca responder à seguinte pergunta:

> É possível prever com alta precisão a demanda energética de curto prazo utilizando informações históricas de consumo, variáveis meteorológicas e modelos de Machine Learning?

Para isso, foram avaliadas diferentes abordagens, incluindo:

- Holt-Winters;
- Random Forest;
- XGBoost;
- LightGBM;
- Voting Regressor;
- Stacking Regressor.

Além da comparação entre modelos, o projeto explora técnicas de Feature Engineering para séries temporais, validação temporal, otimização de hiperparâmetros, análise de generalização e interpretabilidade das previsões.

---

## Dataset

O projeto utiliza o dataset **Tetouan City Power Consumption**, contendo registros de consumo energético e condições meteorológicas ao longo de 2017.

A base original possui:

- **52.416 observações**
- **Frequência:** 10 minutos
- **Início:** 2017-01-01
- **Fim:** 2017-12-30

As variáveis disponíveis originalmente incluem:

```text
DateTime
Temperature
Humidity
Wind Speed
general diffuse flows
diffuse flows
Zone 1 Power Consumption
Zone 2 Power Consumption
Zone 3 Power Consumption
```

Após a padronização:

```text
data_hora
temperatura
umidade
velocidade_vento
fluxo_difuso_geral
fluxo_difuso
consumo_zona_1
consumo_zona_2
consumo_zona_3
```

O projeto concentra a modelagem na variável:

```text
consumo_zona_1
```

As variáveis referentes às Zonas 2 e 3 foram removidas para manter o escopo direcionado à previsão de demanda energética de uma única região.

---

## Estrutura do Projeto

```text
energy-demand-forecasting/
│
├── data/
│   ├── data_energy.csv
│   └── dados_tratados.csv
│
├── models/
│   └── pipeline_final.pkl
│
├── notebooks/
│   ├── 01_validacao_dos_dados.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   └── 03_modelos_ML.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   └── visualization/
│
├── app.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 1. Validação e Preparação dos Dados

Notebook:

`01_validacao_dos_dados.ipynb`

A primeira etapa do projeto foi dedicada à validação da qualidade e consistência da base antes das análises e da modelagem.

Foram realizadas:

- leitura da base original;
- padronização dos nomes das variáveis;
- inspeção da estrutura dos dados;
- validação dos tipos das variáveis;
- verificação de valores ausentes;
- identificação de registros duplicados;
- conversão da variável temporal;
- ordenação cronológica;
- validação da frequência;
- definição do índice temporal;
- análise gráfica inicial;
- exportação da base tratada.

A base apresentou:

```text
52.416 observações
0 valores ausentes
0 registros duplicados
Frequência de 10 minutos
```

A variável `data_hora` foi convertida para `DatetimeIndex`, garantindo uma estrutura adequada para análises temporais.

A base tratada foi então exportada para:

```text
data/dados_tratados.csv
```

---

# 2. Análise Exploratória da Série Temporal

Notebook:

`02_analise_exploratoria.ipynb`

A análise exploratória foi desenvolvida com foco na compreensão da dinâmica temporal do consumo energético da Zona 1.

Foram investigados:

- estatísticas descritivas;
- distribuição do consumo;
- evolução temporal;
- médias móveis;
- sazonalidade mensal;
- comportamento por dia da semana;
- decomposição da série;
- sazonalidade intradiária;
- resíduos;
- estacionariedade;
- ACF;
- PACF;
- diferenciação sazonal;
- diferenciação regular;
- relação com variáveis meteorológicas;
- correlação entre variáveis.

---

## Distribuição do Consumo

O consumo médio observado foi de aproximadamente:

```text
32.345
```

enquanto a mediana foi próxima de:

```text
32.266
```

A proximidade entre média e mediana indica uma distribuição relativamente equilibrada em torno da região central.

Ao mesmo tempo, existe amplitude significativa entre os menores e maiores níveis de consumo, evidenciando variações relevantes de demanda ao longo do período.

<img width="869" height="392" alt="image" src="https://github.com/user-attachments/assets/74de1f0a-a227-4184-9c88-60c0ff62f9b6" />


---

## Evolução Temporal

Para facilitar a visualização do comportamento ao longo do ano, os registros de 10 minutos foram agregados utilizando médias diárias.

A série apresentou mudanças importantes de nível ao longo de 2017.

O consumo aumentou principalmente durante o primeiro semestre, atingindo níveis mais elevados entre junho e agosto.

Nos meses finais do ano ocorreu uma redução mais evidente da demanda.

<img width="1024" height="469" alt="image" src="https://github.com/user-attachments/assets/ab54195e-8d35-4529-8233-6ba526fa75a3" />


---

## Médias Móveis

Foram analisadas médias móveis de:

```text
3 dias
7 dias
```

A média de 3 dias acompanha mais rapidamente as mudanças da série, enquanto a média de 7 dias apresenta um comportamento mais suavizado.

Essas transformações facilitaram a identificação da tendência de consumo ao longo do ano.

<img width="1024" height="471" alt="image" src="https://github.com/user-attachments/assets/115f85fe-9294-4ece-96f3-9ed99f38e153" />


---

## Sazonalidade Mensal

O consumo médio foi analisado por mês.

Os maiores níveis foram observados principalmente entre:

```text
Junho
Julho
Agosto
```

com pico médio próximo ao mês de agosto.

Já novembro e dezembro apresentaram níveis médios mais baixos.

Esse comportamento reforça a existência de uma componente sazonal ao longo do ano.

<img width="1024" height="470" alt="image" src="https://github.com/user-attachments/assets/d378ed51-9228-427e-9e5f-bc06003f5d38" />


---

## Consumo por Dia da Semana

Também foi analisado o comportamento médio do consumo entre os diferentes dias da semana.

O consumo permanece relativamente estável entre segunda-feira e sábado.

No domingo ocorre uma redução mais evidente da demanda média, indicando influência do calendário semanal sobre o padrão energético.

<img width="869" height="501" alt="image" src="https://github.com/user-attachments/assets/775e2b9c-afe1-47e3-96d1-317bd2e618d0" />


---

# 3. Decomposição da Série

A série foi decomposta em:

```text
Série Observada
      ↓
Tendência
      +
Sazonalidade
      +
Resíduo
```

Foi utilizado um modelo aditivo:

```python
seasonal_decompose(
    dados["consumo_zona_1"],
    model="additive",
    period=144
)
```

Como os registros possuem frequência de 10 minutos:

```text
6 observações por hora
24 horas por dia
6 × 24 = 144 observações
```

Portanto, `144` representa um ciclo diário completo.

A decomposição evidenciou:

- mudanças no nível de consumo ao longo do ano;
- forte sazonalidade diária;
- variações residuais não completamente explicadas por tendência e sazonalidade.

<img width="1190" height="985" alt="image" src="https://github.com/user-attachments/assets/12d17430-7636-41b5-bca1-53c93ecf422d" />


---

## Sazonalidade Intradiária

A componente sazonal foi analisada em um recorte de uma semana.

O resultado mostrou um padrão diário claramente repetitivo, com períodos de menor consumo seguidos por elevações pronunciadas de demanda.

A repetição do comportamento ao longo dos dias confirmou a presença de forte sazonalidade intradiária.

<img width="888" height="470" alt="image" src="https://github.com/user-attachments/assets/fd88e593-df7d-4e2e-92d7-f559bb8bce0e" />


---

## Análise dos Resíduos

Os resíduos apresentaram média próxima de zero:

```text
Média ≈ 0.64
```

Porém, a dispersão variou ao longo do período e foram observados valores extremos.

Isso indica que tendência e sazonalidade explicam uma parcela importante da série, mas não capturam completamente toda a dinâmica do consumo energético.

<img width="1036" height="393" alt="image" src="https://github.com/user-attachments/assets/f20947d5-e70c-4a43-81ad-d88731a09a2d" />


---

# 4. Estacionariedade e Autocorrelação

A estacionariedade foi analisada utilizando o teste Augmented Dickey-Fuller.

O resultado da série original apresentou:

```text
ADF Statistic ≈ -32.12
p-value ≈ 0.0000
```

permitindo rejeitar a hipótese de presença de raiz unitária.

Embora o teste indique estacionariedade sob esse critério, isso não significa ausência de sazonalidade ou dependência temporal.

Por esse motivo, também foram analisadas as funções ACF e PACF.

---

## ACF e PACF

A função de autocorrelação apresentou comportamento cíclico bem definido.

Foi observada:

```text
Correlação elevada nos primeiros lags
        ↓
Correlação negativa próxima de 72 lags
        ↓
Nova correlação elevada próxima de 144 lags
```

Como:

```text
72 lags  ≈ 12 horas
144 lags ≈ 24 horas
```

o comportamento reforçou a presença da sazonalidade diária.

<img width="1312" height="449" alt="image" src="https://github.com/user-attachments/assets/bc2c6690-e09f-4861-a4b9-0bc5abaeea6d" />


---

## Diferenciação Sazonal

Foi aplicada diferenciação de 144 períodos:

```python
serie_diff_sazonal = (
    dados["consumo_zona_1"]
    .diff(144)
    .dropna()
)
```

O objetivo foi reduzir a influência do padrão diário.

Apesar da transformação, ainda permaneceu dependência temporal relevante.

---

## Diferenciação Regular + Sazonal

Também foi testada:

```python
serie_diff_completa = (
    dados["consumo_zona_1"]
    .diff()
    .diff(144)
    .dropna()
)
```

Após essa transformação, grande parte das autocorrelações ficou próxima de zero.

Entretanto, como a série original já apresentou evidência de estacionariedade no teste ADF, a diferenciação regular foi interpretada com cautela para evitar sobrediferenciação.

---

# 5. Relação com Variáveis Meteorológicas

Também foram investigadas as relações entre consumo energético e variáveis meteorológicas.

As principais variáveis analisadas foram:

```text
temperatura
umidade
velocidade_vento
fluxo_difuso_geral
fluxo_difuso
```

A matriz de correlação apresentou aproximadamente:

```text
Temperatura       → +0.44
Umidade           → -0.29
Fluxo difuso geral→ +0.19
Velocidade vento  → +0.17
Fluxo difuso      → +0.08
```

A temperatura apresentou a maior associação linear positiva com o consumo.

A umidade apresentou uma relação negativa.

Entretanto, os gráficos de dispersão mostraram elevada variabilidade, indicando que nenhuma variável meteorológica isoladamente é capaz de explicar completamente a demanda.

Esse resultado reforçou a necessidade de combinar:

```text
Variáveis Meteorológicas
          +
Features Temporais
          +
Histórico do Consumo
```

<img width="798" height="393" alt="image" src="https://github.com/user-attachments/assets/c649ceda-1831-4c01-9117-9e8ff811d6dd" />

<img width="1590" height="890" alt="image" src="https://github.com/user-attachments/assets/cf4f770b-dc24-42c4-8b5c-a1aff7b4a67a" />



---

# 6. Preparação para Machine Learning

Modelos tradicionais de Machine Learning não interpretam automaticamente a estrutura temporal da série.

Por esse motivo, o problema foi transformado em uma tarefa de aprendizado supervisionado através de Feature Engineering.

---

## Lags

Foram criadas variáveis representando diferentes horizontes históricos:

```text
lag_1    → 10 minutos
lag_6    → 1 hora
lag_144  → 1 dia
lag_1008 → 1 semana
```

Exemplo:

```python
dados_ml["lag_1"] = dados_ml["consumo_zona_1"].shift(1)
dados_ml["lag_6"] = dados_ml["consumo_zona_1"].shift(6)
dados_ml["lag_144"] = dados_ml["consumo_zona_1"].shift(144)
dados_ml["lag_1008"] = dados_ml["consumo_zona_1"].shift(1008)
```

---

## Médias Móveis

Foram criadas:

```text
rolling_mean_6
rolling_mean_144
```

Para evitar data leakage, as médias móveis utilizam apenas observações anteriores:

```python
dados_ml["rolling_mean_6"] = (
    dados_ml["consumo_zona_1"]
    .shift(1)
    .rolling(6)
    .mean()
)
```

---

## Variáveis Temporais

Também foram criadas informações relacionadas ao calendário:

```text
dia_da_semana
hora_sin
hora_cos
```

A hora do dia foi representada através de seno e cosseno:

```python
dados_ml["hora_sin"] = np.sin(
    2 * np.pi * dados_ml.index.hour / 24
)

dados_ml["hora_cos"] = np.cos(
    2 * np.pi * dados_ml.index.hour / 24
)
```

Essa representação permite preservar a natureza cíclica das horas.

---

# 7. Estratégia de Validação

Os últimos 30 dias foram completamente separados como conjunto de teste final.

A divisão utilizada foi:

```text
Dados de Modelagem
Janeiro ---------------- Novembro

Teste Final
Dezembro
```

O teste final permaneceu isolado durante:

- treinamento inicial;
- tuning;
- comparação;
- seleção de modelos.

Nos dados destinados à modelagem foi utilizado:

```python
TimeSeriesSplit(
    n_splits=5,
    test_size=4320
)
```

Como existem:

```text
6 observações por hora
24 horas por dia
30 dias
```

temos:

```text
6 × 24 × 30 = 4320 observações
```

por janela de validação.

Essa estratégia preserva a ordem cronológica e evita embaralhamento entre passado e futuro.

---

# 8. Métricas de Avaliação

Foram utilizadas três métricas principais.

## MAE — Mean Absolute Error

Representa o erro absoluto médio.

Quanto menor, melhor.

## RMSE — Root Mean Squared Error

Penaliza erros elevados com maior intensidade.

## MAPE — Mean Absolute Percentage Error

Representa o erro percentual médio das previsões.

O **MAE** foi utilizado como principal referência durante as etapas de otimização.

---

# 9. Modelo Estatístico de Referência

O Holt-Winters foi utilizado como benchmark estatístico.

Configuração:

```python
ExponentialSmoothing(
    treino,
    trend="add",
    seasonal="add",
    seasonal_periods=144
)
```

O modelo considera:

```text
Tendência Aditiva
        +
Sazonalidade Aditiva
        +
Ciclo diário de 144 observações
```

Durante a validação temporal, o Holt-Winters apresentou desempenho significativamente inferior aos modelos de Machine Learning.

Esse resultado indicou dificuldade em representar toda a dinâmica da série dentro das janelas de validação adotadas.

---

# 10. Modelos de Machine Learning

Foram inicialmente avaliados:

- Random Forest
- XGBoost
- LightGBM

Todos os modelos utilizaram a mesma estratégia de validação temporal.

Os resultados iniciais mostraram vantagem do LightGBM em relação às configurações iniciais de Random Forest e XGBoost.

---

# 11. Otimização com RandomizedSearchCV

XGBoost e LightGBM foram submetidos à otimização de hiperparâmetros utilizando:

```python
RandomizedSearchCV
```

com:

```python
TimeSeriesSplit
```

e métrica:

```text
neg_mean_absolute_error
```

Foram investigados parâmetros relacionados a:

- quantidade de árvores;
- profundidade;
- learning rate;
- amostragem;
- regularização;
- complexidade das folhas.

Após o tuning, ambos os modelos apresentaram redução significativa dos erros.

---

# 12. Otimização com Optuna

O XGBoost também foi otimizado utilizando Optuna.

Foi utilizado:

```text
TPESampler
```

com:

```python
optuna.samplers.TPESampler(
    seed=42
)
```

A função objetivo utilizou:

```python
cross_val_score(
    scoring="neg_mean_absolute_error"
)
```

junto ao mesmo `TimeSeriesSplit`.

O histórico das tentativas mostrou redução progressiva do melhor MAE.

Os parâmetros de regularização, principalmente:

```text
reg_alpha
reg_lambda
```

apresentaram influência relevante durante o processo de busca.

Entretanto, a configuração encontrada pelo Optuna não superou o melhor resultado obtido anteriormente pelo `RandomizedSearchCV`.

Esse resultado reforçou que uma técnica de otimização mais sofisticada não garante necessariamente melhor desempenho.

<img width="1432" height="436" alt="image" src="https://github.com/user-attachments/assets/2fb7a558-535a-480a-bd5d-8ef7df5dfdd8" />

<img width="1428" height="437" alt="image" src="https://github.com/user-attachments/assets/7f316f9c-f0ac-448d-af41-a1aea658e926" />



---

# 13. Ensemble Learning

Além dos modelos individuais, foram avaliadas estratégias de Ensemble Learning.

Foram testados:

```text
Voting Regressor
Stacking Regressor
```

utilizando como modelos base:

```text
XGBoost otimizado
LightGBM otimizado
```

---

## Voting Regressor

O Voting Regressor combina as previsões dos dois modelos através da média:

```text
XGBoost ──────┐
              ├── Média ──→ Previsão Final
LightGBM ─────┘
```

Sem pesos personalizados, os dois modelos possuem a mesma contribuição.

---

## Stacking Regressor

O Stacking utiliza as previsões dos modelos base como entrada para um estimador final.

A estrutura utilizada foi:

```text
XGBoost ──────┐
              ├── Linear Regression ──→ Previsão
LightGBM ─────┘
```

O Voting Regressor apresentou melhor desempenho médio durante a validação temporal.

---

# 14. Comparação dos Modelos

Os principais resultados médios durante a validação temporal foram:

| Modelo | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Voting Regressor** | **290.91** | **448.58** | **0.88%** |
| XGBoost Otimizado | 295.15 | 455.00 | 0.90% |
| LightGBM Otimizado | 296.27 | 450.79 | 0.90% |
| Stacking Regressor | 297.64 | 452.14 | 0.90% |
| XGBoost Optuna | 298.71 | 455.02 | 0.90% |
| LightGBM | 317.74 | 474.25 | 0.97% |
| Random Forest | 495.38 | 710.28 | 1.47% |
| XGBoost Inicial | 495.38 | 710.28 | 1.47% |

O **Voting Regressor apresentou o melhor desempenho geral**, alcançando os menores valores médios de MAE, RMSE e MAPE.

Isso sugere que a combinação das previsões de XGBoost e LightGBM foi capaz de reduzir parte dos erros individuais dos modelos.

<img width="1594" height="593" alt="image" src="https://github.com/user-attachments/assets/80f5b4da-0a48-43ed-b693-97798bf64721" />


---

# 15. Análise de Generalização

A capacidade de generalização do Voting Regressor foi analisada utilizando Learning Curve.

No maior conjunto analisado:

```text
MAE Treino     ≈ 197.24
MAE Validação  ≈ 353.07
Gap            ≈ 155.83
```

Apesar da diferença entre treino e validação, as curvas não apresentaram divergência crescente.

À medida que novos dados foram adicionados:

```text
Erro de treino
      ↑ moderadamente

Erro de validação
      ↓ progressivamente
```

Esse comportamento não indica overfitting acentuado.

A tendência também sugere que o modelo ainda pode se beneficiar de um volume maior de dados históricos.

<img width="1006" height="546" alt="image" src="https://github.com/user-attachments/assets/8a012395-76b3-4079-9880-174a54b172a3" />


---

# 16. Avaliação Final

Após a seleção do Voting Regressor, o modelo foi treinado utilizando todo o conjunto destinado à modelagem.

Em seguida, foi avaliado sobre o mês reservado exclusivamente como teste final.

Resultados:

```text
MAE  = 208.72
RMSE = 306.24
MAPE = 0.76%
```

O desempenho no teste final foi superior à média observada durante a validação temporal.

Visualmente, as previsões acompanharam de forma muito próxima as oscilações reais do consumo durante dezembro.

<img width="899" height="469" alt="image" src="https://github.com/user-attachments/assets/bbc8310d-7fe8-4edd-a29a-cab7d84daf79" />


---

## Análise dos Resíduos

Os resíduos permaneceram majoritariamente concentrados próximos de zero.

A distribuição apresentou comportamento aproximadamente centrado, embora alguns valores extremos tenham sido identificados.

No gráfico de resíduos versus valores previstos não foi observado um padrão estrutural dominante.

Isso indica ausência de viés sistemático relevante nas previsões.

<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/5d59c6d2-2b7e-4b66-b769-7f9f6ee5a7fa" />


---

# 17. Interpretabilidade com SHAP

Como o modelo final é um `VotingRegressor`, foi utilizado o:

```text
PermutationExplainer
```

permitindo interpretar o ensemble como uma função de previsão completa.

Para reduzir o custo computacional, o SHAP foi calculado sobre uma amostra do conjunto de teste.

---

## Importância Global

A análise mostrou domínio significativo da variável:

```text
lag_1
```

sobre as previsões.

A importância média aproximada foi:

```text
lag_1               → 6708.61
lag_144             → 108.93
fluxo_difuso_geral  → 108.35
rolling_mean_6      → 101.88
rolling_mean_144    → 66.98
hora_cos            → 65.47
hora_sin            → 52.67
lag_1008            → 43.32
lag_6               → 39.61
```

O resultado mostra que o consumo observado 10 minutos antes é a principal informação utilizada pelo modelo.

<img width="794" height="840" alt="image" src="https://github.com/user-attachments/assets/2f990456-9b29-4ca7-a4d6-359eb7581443" />


---

## Dependência do Lag 1

O SHAP Dependence Plot apresentou uma relação positiva extremamente consistente entre:

```text
lag_1
```

e sua contribuição para a previsão.

À medida que o consumo anterior aumenta, sua contribuição SHAP também cresce.

Esse comportamento é coerente com a elevada autocorrelação identificada durante a análise exploratória.

<img width="737" height="490" alt="image" src="https://github.com/user-attachments/assets/b98c46cc-2d1f-418c-9b00-213bfe273379" />


---

## Explicação Individual

O Waterfall Plot foi utilizado para decompor previsões individuais.

Em um dos exemplos analisados:

```text
Valor real     ≈ 23.069
Valor previsto ≈ 23.210
```

O valor médio esperado pelo modelo era aproximadamente:

```text
33.135
```

Entretanto, o `lag_1` apresentou uma contribuição negativa de aproximadamente:

```text
-9.575
```

levando a previsão para uma região muito mais próxima do valor real observado.

A análise reforçou que o modelo utiliza principalmente a persistência de curto prazo da série para produzir suas previsões.

<img width="1033" height="675" alt="image" src="https://github.com/user-attachments/assets/29aa3061-4056-44ed-b735-dc32ab8f5782" />


---

# 18. Pipeline do Modelo

Após a seleção do modelo final, foi construído um pipeline para reunir pré-processamento e previsão em um único artefato.

A estrutura é:

```text
Entrada
   ↓
SimpleImputer
   ↓
Voting Regressor
   ↓
┌───────────────┐
│   XGBoost     │
│   LightGBM    │
└───────────────┘
   ↓
Previsão
```

O `SimpleImputer` utiliza:

```python
SimpleImputer(
    strategy="median"
)
```

garantindo maior robustez caso novas observações apresentem valores ausentes.

O pipeline foi serializado utilizando:

```python
joblib.dump(
    pipeline_final,
    "../models/pipeline_final.pkl"
)
```

O artefato final está localizado em:

```text
models/pipeline_final.pkl
```

Essa estrutura permite carregar diretamente o pipeline em aplicações ou APIs sem reconstruir manualmente os modelos.

---

# 19. Aplicação Streamlit

O projeto inclui uma aplicação desenvolvida com Streamlit para transformar o modelo em uma interface analítica.

A aplicação permite visualizar:

- comportamento histórico da demanda;
- indicadores de consumo;
- projeções de consumo;
- evolução diária;
- períodos de maior demanda;
- períodos de menor demanda;
- comparação entre histórico e projeção;
- informações sobre o modelo.

O dashboard transforma os resultados da modelagem em uma experiência mais próxima de um produto de dados.

<img width="1725" height="858" alt="image" src="https://github.com/user-attachments/assets/71cc5713-fc56-4553-be88-6aee8d862d64" />


---

# 20. Simulação de Demanda Futura

A aplicação também inclui uma simulação de projeção para os próximos 30 dias.

A série histórica e o comportamento projetado são apresentados em conjunto para facilitar a análise visual da evolução esperada da demanda.

É importante destacar que o modelo final foi validado principalmente em um cenário de previsão de curto prazo com atualização contínua das informações históricas.

Como variáveis como:

```text
lag_1
lag_6
lag_144
lag_1008
```

dependem de observações anteriores, uma previsão verdadeiramente realizada com 30 dias completos de antecedência exige uma estratégia recursiva e hipóteses adicionais sobre variáveis futuras.

Da mesma forma, variáveis meteorológicas futuras precisam ser conhecidas ou estimadas.

Portanto, a projeção de 30 dias disponível na aplicação deve ser interpretada como uma **simulação de cenário futuro**, e não como uma previsão operacional com a mesma precisão observada no teste de curto prazo.

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/94472fba-7200-4b91-86a0-922533aace1b" />


---

# Possível Aplicação

Em um cenário real, uma solução desse tipo poderia apoiar:

- monitoramento de demanda energética;
- planejamento operacional;
- gestão de capacidade;
- identificação de períodos de pico;
- planejamento de manutenção;
- otimização do fornecimento;
- apoio à tomada de decisão em operações de energia.

Uma possível arquitetura futura seria:

```text
Novas medições
      ↓
Pipeline de ingestão
      ↓
Validação dos dados
      ↓
Feature Engineering
      ↓
Pipeline de Machine Learning
      ↓
Voting Regressor
      ↓
Previsão de demanda
      ↓
Monitoramento
      ↓
Dashboard / API
      ↓
Tomada de decisão
```

---

# Limitações do Projeto

O desempenho elevado do modelo deve ser interpretado considerando o horizonte de previsão utilizado.

A feature de maior importância é:

```text
lag_1
```

que representa o consumo observado apenas 10 minutos antes.

Portanto, o modelo apresenta excelente desempenho principalmente em cenários de previsão de curtíssimo prazo.

Além disso:

- as variáveis meteorológicas utilizadas precisam estar disponíveis no momento da previsão;
- previsões de longo horizonte exigem tratamento específico das variáveis futuras;
- os valores SHAP representam importância preditiva, e não relações causais;
- o dataset possui apenas aproximadamente um ano de histórico;
- mudanças estruturais futuras no comportamento da demanda não estão representadas nos dados utilizados.

Essas limitações são importantes para evitar interpretações incorretas sobre a capacidade de generalização do modelo.

---

# Principais Tecnologias

O projeto utiliza:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- Streamlit
- Joblib
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# Como Executar o Projeto

Clone o repositório:

```bash
git clone https://github.com/RenanTrevelim/energy-demand-forecasting.git
```

Acesse o diretório:

```bash
cd energy-demand-forecasting
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

---

## Executando os Notebooks

Abra os notebooks na seguinte ordem:

```text
01_validacao_dos_dados.ipynb
02_analise_exploratoria.ipynb
03_modelos_ML.ipynb
```

---

## Executando a Aplicação Streamlit

A partir da raiz do projeto:

```bash
python -m streamlit run app.py
```

A aplicação utiliza o pipeline salvo em:

```text
models/pipeline_final.pkl
```

---

# Fluxo Completo do Projeto

```text
Dados Brutos
      ↓
Validação dos Dados
      ↓
Tratamento e Padronização
      ↓
Análise Exploratória
      ↓
Tendência e Sazonalidade
      ↓
Estacionariedade
      ↓
ACF / PACF
      ↓
Análise Meteorológica
      ↓
Feature Engineering
      ↓
Divisão Temporal
      ↓
TimeSeriesSplit
      ↓
Modelo Estatístico
      +
Machine Learning
      ↓
RandomizedSearchCV
      +
Optuna
      ↓
Voting / Stacking
      ↓
Comparação dos Modelos
      ↓
Voting Regressor
      ↓
Learning Curve
      ↓
Teste Final
      ↓
Análise de Resíduos
      ↓
SHAP
      ↓
Pipeline
      ↓
Serialização
      ↓
Aplicação Streamlit
```

---

# Conclusão

O projeto percorreu as principais etapas de um problema completo de previsão de demanda energética, desde a validação dos dados até a criação de um artefato de Machine Learning reutilizável.

A análise exploratória identificou uma série com forte dependência temporal, sazonalidade diária e variações relevantes ao longo do ano.

A ACF confirmou uma estrutura de autocorrelação associada ao ciclo de 24 horas, enquanto a análise das variáveis meteorológicas mostrou que temperatura e umidade também possuem relação com o consumo.

Para permitir a utilização de modelos de Machine Learning, foram construídas features baseadas em:

```text
Lags
Médias Móveis
Calendário
Representação Cíclica
Variáveis Meteorológicas
```

Os modelos foram avaliados utilizando validação temporal com `TimeSeriesSplit`, mantendo o conjunto final completamente isolado das etapas de seleção.

Após tuning com RandomizedSearchCV, experimentos com Optuna e avaliação de diferentes estratégias de ensemble, o **Voting Regressor**, combinando XGBoost e LightGBM, apresentou o melhor desempenho médio.

Na validação temporal:

```text
MAE  ≈ 290.91
RMSE ≈ 448.58
MAPE ≈ 0.88%
```

No conjunto de teste final:

```text
MAE  = 208.72
RMSE = 306.24
MAPE = 0.76%
```

A análise de resíduos indicou ausência de viés sistemático relevante, enquanto a Learning Curve não apresentou sinais de overfitting acentuado.

A interpretação com SHAP mostrou que `lag_1` é, com grande diferença, a principal variável utilizada pelo modelo.

Esse resultado é coerente com a natureza do problema: em uma série de consumo registrada a cada 10 minutos, o comportamento imediatamente anterior possui elevada capacidade de explicar o próximo nível de demanda.

Por fim, o modelo foi incorporado a um pipeline contendo pré-processamento e ensemble, serializado com Joblib e integrado a uma aplicação Streamlit.

O principal aprendizado do projeto é que uma boa solução de séries temporais não depende apenas da escolha do algoritmo.

Ela exige:

```text
Compreender a série
      ↓
Respeitar sua estrutura temporal
      ↓
Construir features adequadas
      ↓
Validar corretamente
      ↓
Comparar abordagens
      ↓
Interpretar resultados
      ↓
Entender limitações
      ↓
Transformar o modelo em uma solução utilizável
```

---

# Próximas Evoluções

Como próximos passos, o projeto pode ser expandido com:

- transformação do Feature Engineering em componentes reutilizáveis;
- criação de pipeline completo desde os dados brutos;
- integração com previsões meteorológicas futuras;
- avaliação de forecasting multi-step;
- Direct Forecasting para diferentes horizontes;
- comparação com modelos especializados em séries temporais;
- MLflow para rastreamento de experimentos;
- validação automática de schema;
- testes unitários;
- logging;
- Docker;
- API com FastAPI;
- monitoramento de drift;
- atualização automática do modelo;
- CI/CD;
- deploy da aplicação Streamlit;
- deploy em ambiente Cloud;
- integração com fontes de consumo energético em tempo real.

---

# Autor

**Renan Assis Trevelim**

Projeto desenvolvido como aplicação prática de:

- Data Science
- Time Series Analysis
- Machine Learning
- Ensemble Learning
- Feature Engineering
- Hyperparameter Optimization
- Explainable AI
- Model Evaluation
- MLOps
- Streamlit
- Deploy de Modelos

🔗 **LinkedIn:** [Renan Assis Trevelim](https://www.linkedin.com/in/renan-trevelim)

💻 **GitHub:** [RenanTrevelim](https://github.com/RenanTrevelim)

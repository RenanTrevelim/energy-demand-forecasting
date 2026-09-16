import joblib
import pandas as pd

def carregar_modelo(caminho="../models/pipeline_final.pkl"):
    return joblib.load(caminho)

def prever(modelo, X):
    return modelo.predict(X)


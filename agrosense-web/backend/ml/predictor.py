"""
ModelPredictor — carga y ejecuta el bundle de modelos ML
"""
import pickle
import os
import numpy as np
import pandas as pd

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "models", "model_bundle.pkl")

class ModelPredictor:
    def __init__(self):
        self.bundle = None
        self.loaded = False

    def load(self):
        with open(BUNDLE_PATH, "rb") as f:
            self.bundle = pickle.load(f)
        self.loaded = True

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        b = self.bundle
        # Reconstruir variables derivadas
        df = df.copy()
        df["altura_actual"] = df["altura_anterior"] + df["crecimiento_altura"]
        df["copa_actual"]   = df["copa_anterior"]   + df["crecimiento_copa"]

        # Excluir columnas que no son features
        excluded = {"unique_tree_id", "crecimiento_altura", "crecimiento_copa",
                    "monitoreo_label", "id_parcela"}
        feature_cols = [c for c in b["feature_names"] if c in df.columns]

        X = df[feature_cols].copy()

        # Imputar nulls con mediana
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())

        # Codificar categóricas
        for col in b["cat_cols"]:
            if col in X.columns:
                le = b["label_encoders"][col]
                X[col] = X[col].astype(str).apply(
                    lambda v: le.transform([v])[0] if v in le.classes_
                    else le.transform([le.classes_[0]])[0]
                )

        # Asegurar orden correcto de columnas
        X = X[b["feature_names"]]
        return X

    def predict(self, df: pd.DataFrame, model_name: str = "GradientBoosting") -> np.ndarray:
        X = self._preprocess(df)
        model = self.bundle["gb_model"] if model_name == "GradientBoosting" else self.bundle["rf_model"]
        return model.predict(X)

    def detect_anomalies(self, df: pd.DataFrame):
        b = self.bundle
        df2 = df.copy()
        df2["altura_actual"] = df2["altura_anterior"] + df2["crecimiento_altura"]
        df2["copa_actual"]   = df2["copa_anterior"]   + df2["crecimiento_copa"]

        cols = b["iso_cols"]
        X_iso = df2[[c for c in cols if c in df2.columns]].copy()
        for col in X_iso.columns:
            if X_iso[col].isnull().sum() > 0:
                X_iso[col] = X_iso[col].fillna(X_iso[col].median())

        X_scaled = b["iso_scaler"].transform(X_iso)
        preds  = b["iso_model"].predict(X_scaled)
        scores = b["iso_model"].decision_function(X_scaled)
        return preds, scores

    @property
    def feature_importance(self):
        return self.bundle["feature_importance"]

    @property
    def metrics(self):
        return self.bundle["metrics"]

    @property
    def dataset_stats(self):
        return self.bundle["dataset_stats"]

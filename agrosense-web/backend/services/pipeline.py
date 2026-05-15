"""
Pipeline de procesamiento y análisis completo
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def run_pipeline(df: pd.DataFrame, predictor) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo sobre el dataframe:
    1. Feature engineering
    2. Predicciones ML
    3. Detección de anomalías
    4. Estadísticas y resúmenes
    """
    df = df.copy()

    # ── 1. Feature Engineering ──────────────────────────────────
    df["altura_actual"] = df["altura_anterior"] + df["crecimiento_altura"]
    df["copa_actual"]   = df["copa_anterior"]   + df["crecimiento_copa"]

    # ── 2. Predicciones ─────────────────────────────────────────
    preds_gb = predictor.predict(df, model_name="GradientBoosting")
    preds_rf = predictor.predict(df, model_name="RandomForest")
    df["pred_gb"] = preds_gb
    df["pred_rf"] = preds_rf

    # ── 3. Anomalías ─────────────────────────────────────────────
    anomaly_labels, anomaly_scores = predictor.detect_anomalies(df)
    df["anomaly"]       = anomaly_labels
    df["anomaly_score"] = anomaly_scores
    df["is_anomaly"]    = (anomaly_labels == -1)

    # ── 4. Estadísticas generales ────────────────────────────────
    target = "crecimiento_altura"
    general_stats = {
        "n_rows":           int(len(df)),
        "n_trees":          int(df["unique_tree_id"].nunique()),
        "n_species":        int(df["especie"].nunique()),
        "n_anomalies":      int(df["is_anomaly"].sum()),
        "pct_anomalies":    round(float(df["is_anomaly"].mean()) * 100, 1),
        "mean_growth":      round(float(df[target].mean()), 4),
        "median_growth":    round(float(df[target].median()), 4),
        "std_growth":       round(float(df[target].std()), 4),
        "min_growth":       round(float(df[target].min()), 4),
        "max_growth":       round(float(df[target].max()), 4),
        "pct_zero_growth":  round(float((df[target] == 0).mean()) * 100, 1),
        "pct_neg_growth":   round(float((df[target] < 0).mean()) * 100, 1),
    }

    # ── 5. Crecimiento por especie ───────────────────────────────
    by_species = (
        df.groupby("especie")[target]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_growth", "std": "std_growth", "count": "n"})
        .reset_index()
        .sort_values("mean_growth", ascending=False)
    )
    by_species["mean_growth"] = by_species["mean_growth"].round(4)
    by_species["std_growth"]  = by_species["std_growth"].fillna(0).round(4)

    # ── 6. Crecimiento por localidad ─────────────────────────────
    loc_map = {
        "loc_Guayabal": "Guayabal",
        "loc_San Antonio": "San Antonio",
        "loc_Tres Jotas": "Tres Jotas",
    }
    by_location = []
    for col, name in loc_map.items():
        if col in df.columns:
            subset = df[df[col] == 1]
            by_location.append({
                "localidad":    name,
                "mean_growth":  round(float(subset[target].mean()), 4) if len(subset) > 0 else 0,
                "std_growth":   round(float(subset[target].std()), 4) if len(subset) > 0 else 0,
                "n":            int(len(subset)),
            })

    # ── 7. Crecimiento por monitoreo ─────────────────────────────
    by_monitoreo = (
        df.groupby("monitoreo")[target]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_growth", "std": "std_growth", "count": "n"})
        .reset_index()
    )
    by_monitoreo["mean_growth"] = by_monitoreo["mean_growth"].round(4)
    by_monitoreo["std_growth"]  = by_monitoreo["std_growth"].fillna(0).round(4)

    # ── 8. Distribución del target (histograma) ──────────────────
    counts, bin_edges = np.histogram(df[target].dropna(), bins=30)
    distribution = [
        {"bin": round(float(bin_edges[i]), 3), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    # ── 9. Tabla de anomalías ────────────────────────────────────
    anomaly_cols = ["unique_tree_id", "monitoreo", "especie", "crecimiento_altura",
                    "altura_anterior", "altura_actual", "copa_actual", "anomaly_score"]
    anomaly_cols = [c for c in anomaly_cols if c in df.columns]
    anomaly_df = (
        df[df["is_anomaly"]][anomaly_cols]
        .sort_values("anomaly_score")
        .head(50)
    )
    anomaly_df["anomaly_score"] = anomaly_df["anomaly_score"].round(4)
    anomaly_df["crecimiento_altura"] = anomaly_df["crecimiento_altura"].round(4)

    # ── 10. Dataset procesado (primeras 500 filas para la tabla) ──
    output_cols = ["unique_tree_id", "monitoreo", "especie", "localidad_str",
                   "altura_anterior", "altura_actual", "copa_anterior", "copa_actual",
                   "crecimiento_altura", "pred_gb", "is_anomaly"]
    df["localidad_str"] = np.select(
        [df.get("loc_Guayabal", pd.Series(0)) == 1,
         df.get("loc_San Antonio", pd.Series(0)) == 1,
         df.get("loc_Tres Jotas", pd.Series(0)) == 1],
        ["Guayabal", "San Antonio", "Tres Jotas"],
        default="Desconocida"
    )
    avail_cols = [c for c in output_cols if c in df.columns]
    table_df = df[avail_cols].copy().head(500)
    for col in ["altura_anterior", "altura_actual", "copa_anterior", "copa_actual",
                "crecimiento_altura", "pred_gb"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(4)

    return {
        "general_stats":   general_stats,
        "by_species":      by_species.to_dict(orient="records"),
        "by_location":     by_location,
        "by_monitoreo":    by_monitoreo.to_dict(orient="records"),
        "distribution":    distribution,
        "anomalies":       anomaly_df.fillna("").to_dict(orient="records"),
        "table":           table_df.fillna("").to_dict(orient="records"),
        "feature_importance": predictor.feature_importance,
        "model_metrics":   predictor.metrics,
    }

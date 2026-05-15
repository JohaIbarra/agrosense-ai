"""
Validaciones del dataset de entrada
"""
import pandas as pd
from typing import Tuple, List

REQUIRED_COLUMNS = [
    "unique_tree_id", "monitoreo", "especie", "familia", "nombre_comun",
    "meses_desde_inicio", "elevacion_m", "coord_x", "coord_y",
    "altura_anterior", "copa_anterior",
    "crecimiento_altura", "crecimiento_copa",
    "fito_Bueno", "fito_Malo", "fito_Regular",
    "gremio_Inicial", "gremio_Intermedia",
    "loc_Guayabal", "loc_San Antonio", "loc_Tres Jotas",
]

NUMERIC_COLUMNS = [
    "monitoreo", "meses_desde_inicio", "elevacion_m", "coord_x", "coord_y",
    "altura_anterior", "copa_anterior", "crecimiento_altura", "crecimiento_copa",
    "fito_Bueno", "fito_Malo", "fito_Regular",
    "gremio_Inicial", "gremio_Intermedia",
    "loc_Guayabal", "loc_San Antonio", "loc_Tres Jotas",
]

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []

    # Verificar columnas requeridas
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Columnas faltantes: {', '.join(missing)}")

    if errors:
        return False, errors

    # Verificar tipos numéricos
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception:
                errors.append(f"Columna '{col}' contiene valores no numéricos")

    # Verificar columnas críticas sin nulls
    critical = ["unique_tree_id", "altura_anterior", "crecimiento_altura"]
    for col in critical:
        if col in df.columns and df[col].isnull().sum() > 0:
            n = df[col].isnull().sum()
            errors.append(f"Columna crítica '{col}' tiene {n} valores nulos")

    # Verificar valores lógicos
    if "altura_anterior" in df.columns:
        neg = (pd.to_numeric(df["altura_anterior"], errors='coerce') < 0).sum()
        if neg > 0:
            errors.append(f"'altura_anterior' tiene {neg} valores negativos")

    return len(errors) == 0, errors

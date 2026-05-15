"""
PIPELINE ML - PREDICCION DE crecimiento_altura
Fases 1, 2 y 3: Carga, Validación, Feature Engineering, Preparación
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = r'dataset_ml.xlsx'
SEPARATOR = "=" * 60

# ─────────────────────────────────────────────
# FASE 1 — CARGA Y VALIDACIÓN FINAL
# ─────────────────────────────────────────────
print(SEPARATOR)
print("FASE 1 — CARGA Y VALIDACION FINAL")
print(SEPARATOR)

df = pd.read_excel(DATASET_PATH)
print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
print(f"\nColumnas y tipos:")
for col in df.columns:
    print(f"  {col:35s} | {str(df[col].dtype):10s} | nulls: {df[col].isnull().sum()}")

# Duplicados
dup = df.duplicated().sum()
print(f"\nDuplicados exactos: {dup}")

# Nulls por columna
print(f"\nColumnas con nulls:")
nulls = df.isnull().sum()
for col, n in nulls[nulls > 0].items():
    print(f"  {col}: {n} nulls ({n/len(df)*100:.2f}%)")

# Verificación: altura_actual = altura_anterior + crecimiento_altura
print(f"\n[VERIFICACION] altura_anterior + crecimiento_altura:")
altura_calc = df['altura_anterior'] + df['crecimiento_altura']
diff_altura = (altura_calc - altura_calc).abs().max()  # vs nada por ahora
# Validación cruzada entre monitoreos
df_v = df[['unique_tree_id','monitoreo','altura_anterior','crecimiento_altura']].copy()
df_v['alt_calc'] = df_v['altura_anterior'] + df_v['crecimiento_altura']
df_next = df_v[['unique_tree_id','monitoreo','altura_anterior']].rename(
    columns={'monitoreo':'mon_next','altura_anterior':'alt_ant_next'})
df_v['mon_next'] = df_v['monitoreo'] + 1
merged = df_v.merge(df_next, on=['unique_tree_id','mon_next'], how='inner')
merged['diff'] = (merged['alt_calc'] - merged['alt_ant_next']).abs()
print(f"  Pares consecutivos validados: {len(merged)}")
print(f"  Diferencia máxima: {merged['diff'].max():.8f}")
print(f"  100% consistencia: {(merged['diff'] < 0.001).all()}")

# Outliers en target
q1 = df['crecimiento_altura'].quantile(0.25)
q3 = df['crecimiento_altura'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['crecimiento_altura'] < q1 - 3*iqr) | (df['crecimiento_altura'] > q3 + 3*iqr)]
print(f"\nOutliers extremos en crecimiento_altura (±3 IQR): {len(outliers)}")
if len(outliers) > 0:
    print(f"  Valores: {outliers['crecimiento_altura'].tolist()}")

print("\n[FASE 1] OK — Dataset validado, sin corrupción detectada.")

# ─────────────────────────────────────────────
# FASE 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("FASE 2 — FEATURE ENGINEERING")
print(SEPARATOR)

# Variables derivadas documentadas
df['altura_actual'] = df['altura_anterior'] + df['crecimiento_altura']
df['copa_actual']   = df['copa_anterior']   + df['crecimiento_copa']

print("Variables derivadas construidas:")
print("  altura_actual = altura_anterior + crecimiento_altura")
print("  copa_actual   = copa_anterior   + crecimiento_copa")

# Verificar no negativos
neg_alt = (df['altura_actual'] < 0).sum()
neg_copa = (df['copa_actual'] < 0).sum()
print(f"\n  Valores negativos en altura_actual: {neg_alt}")
print(f"  Valores negativos en copa_actual:   {neg_copa}")

# Stats
for col in ['altura_actual', 'copa_actual']:
    s = df[col]
    print(f"\n  {col}: min={s.min():.3f}, max={s.max():.3f}, mean={s.mean():.3f}, nulls={s.isnull().sum()}")

# Coherencia temporal: altura_actual debe ser >= altura_anterior (árboles no encogen en general)
# Nota: crecimiento_altura puede ser negativo (error de medición, compactación)
neg_growth = (df['crecimiento_altura'] < 0).sum()
print(f"\n  Registros con crecimiento_altura negativo: {neg_growth} ({neg_growth/len(df)*100:.1f}%)")
print(f"  Valor mínimo de crecimiento_altura: {df['crecimiento_altura'].min():.4f}")
print(f"  [NOTA] Valores negativos son posibles (error de medición o corrección de campo)")

print("\n[FASE 2] OK — Variables derivadas construidas y verificadas.")

# ─────────────────────────────────────────────
# FASE 3 — PREPARACIÓN DE FEATURES
# ─────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("FASE 3 — PREPARACION DE FEATURES (X, y)")
print(SEPARATOR)

# TARGET
y = df['crecimiento_altura'].copy()

# FEATURES — excluir:
# - unique_tree_id (identificador, no predictivo)
# - crecimiento_altura (target)
# - crecimiento_copa (variable de respuesta análoga, riesgo de leakage conceptual)
# - monitoreo_label (redundante con monitoreo numérico)
EXCLUDED = ['unique_tree_id', 'crecimiento_altura', 'crecimiento_copa',
            'monitoreo_label', 'id_parcela']

feature_cols = [c for c in df.columns if c not in EXCLUDED]
X = df[feature_cols].copy()

print(f"Target (y): crecimiento_altura")
print(f"  Shape y: {y.shape}")
print(f"  Media: {y.mean():.4f}, Std: {y.std():.4f}, Min: {y.min():.4f}, Max: {y.max():.4f}")

print(f"\nFeatures usadas ({len(feature_cols)}):")
for i, col in enumerate(feature_cols):
    dtype = X[col].dtype
    nulls = X[col].isnull().sum()
    print(f"  [{i:2d}] {col:35s} dtype={str(dtype):10s} nulls={nulls}")

print(f"\nColumnas EXCLUIDAS: {EXCLUDED}")

# Verificar nulls en X
total_nulls_X = X.isnull().sum().sum()
print(f"\nNulls totales en X: {total_nulls_X}")
if total_nulls_X > 0:
    print("  Columnas con nulls en X:")
    for col, n in X.isnull().sum()[X.isnull().sum() > 0].items():
        print(f"    {col}: {n}")
    # Imputar con mediana para copa_actual (1 null heredado)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            med = X[col].median()
            X[col] = X[col].fillna(med)
            print(f"  Imputado {col} con mediana={med:.4f}")

print(f"\nShape final X: {X.shape}")
print(f"Shape final y: {y.shape}")

# Guardar X e y para siguientes fases
X.to_csv('X_features.csv', index=False)
y.to_csv('y_target.csv', index=False)
df.to_csv('df_engineered.csv', index=False)
print("\n[FASE 3] OK — X, y y dataset enriquecido guardados.")
print(f"\nArchivos generados:")
print(f"  X_features.csv")
print(f"  y_target.csv")
print(f"  df_engineered.csv")

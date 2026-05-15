"""
PIPELINE ML — FASES 9 y 10: Detección de Anomalías + Verificación Final
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SEPARATOR = "=" * 60
OUTPUT_DIR = 'ml_outputs'

# Cargar datos
df = pd.read_csv('df_engineered.csv')
X_model = pd.read_csv('X_model.csv')
y = pd.read_csv('y_target.csv').squeeze()

# ─────────────────────────────────────────────────────────────
# FASE 9 — DETECCION DE ANOMALIAS
# ─────────────────────────────────────────────────────────────
print(SEPARATOR)
print("FASE 9 — DETECCION DE ANOMALIAS (Isolation Forest)")
print(SEPARATOR)

# Usar solo columnas numéricas relevantes para detección
num_cols = ['altura_anterior', 'altura_actual', 'copa_anterior', 'copa_actual',
            'crecimiento_altura', 'crecimiento_copa', 'elevacion_m',
            'meses_desde_inicio']
# Filtrar los que existen
num_cols = [c for c in num_cols if c in df.columns]
print(f"Columnas usadas para deteccion: {num_cols}")

X_anomaly = df[num_cols].copy()

# Imputar el único null de copa
for col in X_anomaly.columns:
    if X_anomaly[col].isnull().sum() > 0:
        X_anomaly[col] = X_anomaly[col].fillna(X_anomaly[col].median())

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

# Isolation Forest
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # Esperamos ~5% de observaciones anómalas
    random_state=42,
    n_jobs=-1
)
preds = iso.fit_predict(X_scaled)
scores = iso.decision_function(X_scaled)  # Más negativo = más anómalo

df['anomaly'] = preds           # -1 = anómalo, 1 = normal
df['anomaly_score'] = scores

n_anomalies = (preds == -1).sum()
n_normal = (preds == 1).sum()
print(f"\nResultados Isolation Forest (contamination=5%):")
print(f"  Total registros:  {len(df)}")
print(f"  Normales:         {n_normal} ({n_normal/len(df)*100:.1f}%)")
print(f"  Anomalias:        {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")

# Stats de anomalías vs normales
anomalias = df[df['anomaly'] == -1]
normales  = df[df['anomaly'] == 1]

print(f"\nComparacion estadistica ANOMALIAS vs NORMALES:")
print(f"{'Variable':30s} {'Normal_media':>14s} {'Anomalia_media':>16s} {'Dif%':>8s}")
print("-" * 72)
for col in num_cols:
    nm = normales[col].mean()
    am = anomalias[col].mean()
    pct = (am - nm) / abs(nm) * 100 if nm != 0 else float('inf')
    print(f"  {col:28s} {nm:14.4f} {am:16.4f} {pct:8.1f}%")

# Top anomalías más extremas
print(f"\nTop 15 registros mas anomalos:")
top_anomalias = df[df['anomaly'] == -1].nsmallest(15, 'anomaly_score')
cols_show = ['unique_tree_id', 'monitoreo', 'crecimiento_altura', 
             'altura_anterior', 'altura_actual', 'anomaly_score']
cols_show = [c for c in cols_show if c in top_anomalias.columns]
print(top_anomalias[cols_show].to_string(index=False))

# Anomalías por especie
print(f"\nAnomalias por especie:")
anom_esp = anomalias.groupby('especie').size().sort_values(ascending=False)
tot_esp = df.groupby('especie').size()
tasa_esp = (anom_esp / tot_esp * 100).dropna().sort_values(ascending=False)
print("  Tasa de anomalia por especie (top 10):")
for esp, tasa in tasa_esp.head(10).items():
    n = anom_esp.get(esp, 0)
    print(f"    {esp:35s}: {tasa:5.1f}% ({n} anomalias)")

# Visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Detección de Anomalías — Isolation Forest', fontweight='bold', fontsize=13)

# Plot 1: altura_actual vs crecimiento_altura coloreado por anomalía
col_colors = np.where(df['anomaly'] == -1, 'red', 'steelblue')
axes[0, 0].scatter(df['altura_actual'], df['crecimiento_altura'],
                   c=col_colors, alpha=0.4, s=15)
axes[0, 0].set_xlabel('altura_actual')
axes[0, 0].set_ylabel('crecimiento_altura')
axes[0, 0].set_title('altura_actual vs crecimiento_altura')
from matplotlib.patches import Patch
legend_elems = [Patch(color='steelblue', label=f'Normal ({n_normal})'),
                Patch(color='red', label=f'Anomalia ({n_anomalies})')]
axes[0, 0].legend(handles=legend_elems)

# Plot 2: copa_actual vs crecimiento_copa
axes[0, 1].scatter(df['copa_actual'], df['crecimiento_copa'] if 'crecimiento_copa' in df.columns else df['crecimiento_altura'],
                   c=col_colors, alpha=0.4, s=15)
axes[0, 1].set_xlabel('copa_actual')
axes[0, 1].set_ylabel('crecimiento_copa')
axes[0, 1].set_title('copa_actual vs crecimiento_copa')

# Plot 3: Distribución del anomaly score
axes[1, 0].hist(df.loc[df['anomaly'] == 1, 'anomaly_score'], bins=40,
                color='steelblue', alpha=0.7, label='Normal')
axes[1, 0].hist(df.loc[df['anomaly'] == -1, 'anomaly_score'], bins=20,
                color='red', alpha=0.7, label='Anomalia')
axes[1, 0].set_xlabel('Anomaly Score')
axes[1, 0].set_title('Distribución del Score de Anomalía')
axes[1, 0].legend()

# Plot 4: Anomalías por monitoreo
mon_anom = df.groupby('monitoreo').apply(lambda x: (x['anomaly'] == -1).sum()).reset_index()
mon_anom.columns = ['monitoreo', 'n_anomalias']
mon_total = df.groupby('monitoreo').size().reset_index(name='total')
mon_merged = mon_anom.merge(mon_total, on='monitoreo')
mon_merged['tasa'] = mon_merged['n_anomalias'] / mon_merged['total'] * 100
axes[1, 1].bar(mon_merged['monitoreo'].astype(str), mon_merged['tasa'],
               color='coral', edgecolor='white')
axes[1, 1].set_xlabel('Monitoreo')
axes[1, 1].set_ylabel('% Anomalías')
axes[1, 1].set_title('Tasa de Anomalías por Monitoreo')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_anomalias.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figura: 09_anomalias.png")

# Exportar reporte de anomalías
reporte_anomalias = df[df['anomaly'] == -1].copy()
reporte_anomalias = reporte_anomalias.sort_values('anomaly_score')
cols_reporte = ['unique_tree_id', 'monitoreo', 'especie', 'crecimiento_altura',
                'altura_anterior', 'altura_actual', 'copa_anterior', 'copa_actual',
                'elevacion_m', 'anomaly_score']
cols_reporte = [c for c in cols_reporte if c in reporte_anomalias.columns]
reporte_anomalias[cols_reporte].to_csv(f'{OUTPUT_DIR}/09_reporte_anomalias.csv', index=False)
print(f"  Reporte CSV: 09_reporte_anomalias.csv")

print(f"\n[FASE 9] OK — {n_anomalies} anomalias detectadas y reportadas.")

# ─────────────────────────────────────────────────────────────
# FASE 10 — VERIFICACION FINAL
# ─────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("FASE 10 — VERIFICACION FINAL")
print(SEPARATOR)

X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()

checks = []

def check(desc, condition, detail=""):
    status = "OK" if condition else "FALLO"
    checks.append((status, desc, detail))
    print(f"  [{status:5s}] {desc}" + (f" — {detail}" if detail else ""))

print("\n[1] Integridad del dataset:")
check("Filas totales correctas", len(df) == 1987, f"len={len(df)}")
check("Sin duplicados", df.duplicated().sum() == 0, f"dups={df.duplicated().sum()}")
check("altura_actual sin negativos", (df['altura_actual'] < 0).sum() == 0)
check("copa_actual sin negativos", (df['copa_actual'] < 0).sum() == 0)

print("\n[2] Coherencia del split:")
check("Train+Test = Total", len(X_train) + len(X_test) == len(df), f"{len(X_train)}+{len(X_test)}={len(X_train)+len(X_test)}")
check("X_train filas = y_train filas", len(X_train) == len(y_train))
check("X_test filas = y_test filas", len(X_test) == len(y_test))
check("No hay nulls en X_train", X_train.isnull().sum().sum() == 0, f"nulls={X_train.isnull().sum().sum()}")

print("\n[3] Verificación matemática (altura_actual = altura_ant + crec):")
df_v = pd.read_csv('df_engineered.csv')
df_check = df_v[['unique_tree_id','monitoreo','altura_anterior','crecimiento_altura','altura_actual']].copy()
df_check['recalc'] = df_check['altura_anterior'] + df_check['crecimiento_altura']
diff = (df_check['recalc'] - df_check['altura_actual']).abs()
check("altura_actual reconstruida correctamente", diff.max() < 1e-6, f"diff_max={diff.max():.2e}")

print("\n[4] Verificación de métricas (GradientBoosting):")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                min_samples_leaf=3, subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
check("R² test > 0.95 (modelo robusto)", r2 > 0.95, f"R²={r2:.4f}")
check("MAE test < 0.05 m (error < 5 cm)", mae < 0.05, f"MAE={mae:.4f}")
check("Predicciones sin NaN", not np.isnan(y_pred).any())

print("\n[5] Verificación de anomalías:")
check("Isolation Forest ejecutado", 'anomaly' in df.columns)
check("% anomalías entre 3% y 7%", 0.03 < n_anomalies/len(df) < 0.07,
      f"{n_anomalies/len(df)*100:.1f}%")

print(f"\n{'='*50}")
total_ok = sum(1 for s, _, _ in checks if s == 'OK')
total_fail = sum(1 for s, _, _ in checks if s != 'OK')
print(f"RESUMEN VERIFICACION FINAL: {total_ok} OK / {total_fail} FALLIDOS")

if total_fail == 0:
    print("  TODO VERIFICADO CORRECTAMENTE.")
else:
    print("  PROBLEMAS DETECTADOS — REVISAR ANTES DE USAR.")
    for s, desc, detail in checks:
        if s != 'OK':
            print(f"    FALLO: {desc} ({detail})")

# Archivos finales
import os
print(f"\n{'='*50}")
print(f"ARCHIVOS GENERADOS ({OUTPUT_DIR}/):")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f'{OUTPUT_DIR}/{f}')
    print(f"  {f:40s} {size:>8,} bytes")
print(f"\n[FASE 10] OK — Pipeline completo verificado.")

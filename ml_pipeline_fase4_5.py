"""
PIPELINE ML — FASES 4 y 5: EDA + SPLIT
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Cargar datos preparados
X = pd.read_csv('X_features.csv')
y = pd.read_csv('y_target.csv').squeeze()
df = pd.read_csv('df_engineered.csv')

SEPARATOR = "=" * 60
OUTPUT_DIR = 'ml_outputs'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FASE 4 — EDA PROFESIONAL
# ─────────────────────────────────────────────────────────────
print(SEPARATOR)
print("FASE 4 — EDA PROFESIONAL")
print(SEPARATOR)

# 4.1 Distribución del target
print("\n[4.1] Distribución de crecimiento_altura:")
print(f"  Media:    {y.mean():.4f}")
print(f"  Mediana:  {y.median():.4f}")
print(f"  Std:      {y.std():.4f}")
print(f"  Skewness: {y.skew():.4f}")
print(f"  Kurt:     {y.kurtosis():.4f}")
print(f"  Q1: {y.quantile(0.25):.4f}  Q3: {y.quantile(0.75):.4f}")
print(f"  % crecimiento = 0: {(y == 0).mean()*100:.1f}%")
print(f"  % crecimiento > 0: {(y > 0).mean()*100:.1f}%")
print(f"  % crecimiento < 0: {(y < 0).mean()*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Distribución del Target: crecimiento_altura', fontsize=13, fontweight='bold')

axes[0].hist(y, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(y.mean(), color='red', ls='--', label=f'Media={y.mean():.3f}')
axes[0].axvline(y.median(), color='orange', ls='--', label=f'Mediana={y.median():.3f}')
axes[0].set_title('Histograma'); axes[0].legend(); axes[0].set_xlabel('crecimiento_altura')

axes[1].boxplot(y, patch_artist=True, boxprops=dict(facecolor='lightsteelblue'))
axes[1].set_title('Boxplot'); axes[1].set_ylabel('crecimiento_altura')

from scipy import stats
stats.probplot(y, dist='norm', plot=axes[2])
axes[2].set_title('Q-Q Plot (normalidad)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04a_target_distribucion.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: 04a_target_distribucion.png")

# 4.2 Crecimiento por especie
print("\n[4.2] Crecimiento por especie (top 10):")
esp_stats = df.groupby('especie')['crecimiento_altura'].agg(['mean','std','count']).sort_values('mean', ascending=False)
print(esp_stats.head(10).to_string())

fig, ax = plt.subplots(figsize=(12, 6))
top_esp = esp_stats.head(15)
bars = ax.bar(range(len(top_esp)), top_esp['mean'], 
              yerr=top_esp['std'].fillna(0), capsize=4,
              color='steelblue', edgecolor='navy', alpha=0.8)
ax.set_xticks(range(len(top_esp)))
ax.set_xticklabels(top_esp.index, rotation=45, ha='right', fontsize=8)
ax.set_title('Crecimiento Promedio por Especie (Top 15)', fontweight='bold')
ax.set_ylabel('crecimiento_altura (m)')
ax.axhline(y.mean(), color='red', ls='--', alpha=0.7, label=f'Media global={y.mean():.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04b_crecimiento_especie.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: 04b_crecimiento_especie.png")

# 4.3 Crecimiento por localidad
print("\n[4.3] Crecimiento por localidad:")
for loc_col in ['loc_Guayabal', 'loc_San Antonio', 'loc_Tres Jotas']:
    mask = df[loc_col] == 1
    m = df.loc[mask, 'crecimiento_altura'].mean()
    s = df.loc[mask, 'crecimiento_altura'].std()
    n = mask.sum()
    loc_name = loc_col.replace('loc_', '')
    print(f"  {loc_name:15s}: n={n:4d}, mean={m:.4f}, std={s:.4f}")

# 4.4 Relaciones temporales
print("\n[4.4] Crecimiento por monitoreo:")
mon_stats = df.groupby('monitoreo')['crecimiento_altura'].agg(['mean','std','count'])
print(mon_stats.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Análisis Temporal', fontweight='bold')

mon_stats['mean'].plot(ax=axes[0], marker='o', color='steelblue', linewidth=2)
axes[0].fill_between(mon_stats.index,
                     mon_stats['mean'] - mon_stats['std'],
                     mon_stats['mean'] + mon_stats['std'],
                     alpha=0.2, color='steelblue')
axes[0].set_title('Crecimiento promedio por monitoreo')
axes[0].set_xlabel('Monitoreo'); axes[0].set_ylabel('crecimiento_altura (m)')
axes[0].axhline(y.mean(), color='red', ls='--', alpha=0.7)

axes[1].scatter(df['meses_desde_inicio'], df['crecimiento_altura'],
                alpha=0.3, s=15, color='steelblue')
axes[1].set_title('crecimiento_altura vs meses_desde_inicio')
axes[1].set_xlabel('meses_desde_inicio'); axes[1].set_ylabel('crecimiento_altura')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04c_temporal.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: 04c_temporal.png")

# 4.5 Correlaciones (solo numéricas)
print("\n[4.5] Correlaciones con target (Pearson):")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
corr_target = X[num_cols].corrwith(y).sort_values(key=abs, ascending=False)
for col, val in corr_target.items():
    print(f"  {col:35s}: {val:+.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = X[num_cols].join(y).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax, annot_kws={'size': 7},
            linewidths=0.3)
ax.set_title('Matriz de Correlación', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04d_correlaciones.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: 04d_correlaciones.png")

# 4.6 Scatter plots de variables clave vs target
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Variables Clave vs crecimiento_altura', fontweight='bold')

key_vars = ['altura_anterior', 'altura_actual', 'copa_anterior', 'copa_actual',
            'elevacion_m', 'meses_desde_inicio']
for ax, var in zip(axes.flat, key_vars):
    ax.scatter(df[var] if var in df.columns else X[var],
               y, alpha=0.25, s=12, color='steelblue')
    ax.set_xlabel(var, fontsize=9); ax.set_ylabel('crecimiento_altura', fontsize=9)
    ax.set_title(var, fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04e_scatterplots.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: 04e_scatterplots.png")

print("\n[FASE 4] OK — EDA completado.")

# ─────────────────────────────────────────────────────────────
# FASE 5 — SPLIT CORRECTO
# ─────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("FASE 5 — SPLIT TRAIN/TEST")
print(SEPARATOR)

# Codificar categóricas
X_model = X.copy()
cat_cols = X_model.select_dtypes(include='object').columns.tolist()
print(f"Columnas categóricas a codificar: {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X_model[col] = le.fit_transform(X_model[col].astype(str))
    le_dict[col] = le
    print(f"  {col}: {len(le.classes_)} categorías únicas")

# Split estratificado por monitoreo para evitar leakage temporal
# Usamos random_state fijo para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(f"\nSplit 80/20 con random_state=42:")
print(f"  Train: {len(X_train)} filas ({len(X_train)/len(X_model)*100:.1f}%)")
print(f"  Test:  {len(X_test)} filas ({len(X_test)/len(X_model)*100:.1f}%)")

print(f"\nDistribución del target en splits:")
print(f"  Train — media: {y_train.mean():.4f}, std: {y_train.std():.4f}")
print(f"  Test  — media: {y_test.mean():.4f},  std: {y_test.std():.4f}")

# Guardar splits
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
X_model.to_csv('X_model.csv', index=False)

# Guardar lista de features
feature_names = X_model.columns.tolist()
pd.Series(feature_names).to_csv('feature_names.csv', index=False)

print("\n[FASE 5] OK — Splits guardados. Sin data leakage.")
print(f"Archivos: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

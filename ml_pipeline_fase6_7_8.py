"""
PIPELINE ML — FASES 6, 7 y 8: Entrenamiento, Evaluación, Feature Importance
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

SEPARATOR = "=" * 60
OUTPUT_DIR = 'ml_outputs'

# Cargar splits
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()
feature_names = pd.read_csv('feature_names.csv').squeeze().tolist()

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# ─────────────────────────────────────────────────────────────
# FASE 6 — ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("FASE 6 — ENTRENAMIENTO DE MODELOS")
print(SEPARATOR)

models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
}

# Intentar XGBoost
try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    print("XGBoost disponible. Incluido en comparacion.")
except ImportError:
    print("XGBoost no instalado. Se omite.")

results = {}

for name, model in models.items():
    print(f"\n  Entrenando {name}...")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    
    # Cross-validation en train
    cv_scores = cross_val_score(model, X_train, y_train, 
                                 cv=5, scoring='r2', n_jobs=-1)
    
    results[name] = {
        'model': model,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train,
        'MAE_train':  mean_absolute_error(y_train, y_pred_train),
        'RMSE_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'R2_train':   r2_score(y_train, y_pred_train),
        'MAE_test':   mean_absolute_error(y_test, y_pred_test),
        'RMSE_test':  np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R2_test':    r2_score(y_test, y_pred_test),
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std':  cv_scores.std(),
    }
    print(f"  {name} OK.")

print(f"\n[FASE 6] OK — {len(models)} modelos entrenados.")

# ─────────────────────────────────────────────────────────────
# FASE 7 — EVALUACION
# ─────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("FASE 7 — EVALUACION Y METRICAS")
print(SEPARATOR)

print(f"\n{'Modelo':25s} {'MAE_tr':>8s} {'RMSE_tr':>8s} {'R2_tr':>7s} | {'MAE_te':>8s} {'RMSE_te':>8s} {'R2_te':>7s} | {'CV_R2':>7s} {'±':>6s}")
print("-" * 105)
for name, r in results.items():
    print(f"{name:25s} {r['MAE_train']:8.4f} {r['RMSE_train']:8.4f} {r['R2_train']:7.4f} | "
          f"{r['MAE_test']:8.4f} {r['RMSE_test']:8.4f} {r['R2_test']:7.4f} | "
          f"{r['CV_R2_mean']:7.4f} {r['CV_R2_std']:6.4f}")

# Interpretación honesta
print(f"\n[INTERPRETACION HONESTA]")
best_name = max(results, key=lambda k: results[k]['R2_test'])
best = results[best_name]
print(f"  Mejor modelo en test: {best_name}")
print(f"  R²={best['R2_test']:.4f} → explica {best['R2_test']*100:.1f}% de la varianza")
print(f"  MAE={best['MAE_test']:.4f} m → error promedio de {best['MAE_test']*100:.1f} cm")
print(f"  RMSE={best['RMSE_test']:.4f} m")

# Comparar R2 train vs test para detectar overfitting
gap = best['R2_train'] - best['R2_test']
print(f"  Gap R2 (train-test): {gap:.4f}", end="")
if gap > 0.15:
    print(f"  [ALERTA: posible sobreajuste]")
elif gap > 0.08:
    print(f"  [MODERADO: revisar regularizacion]")
else:
    print(f"  [OK: sin sobreajuste significativo]")

# Plots de evaluación
fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
if len(results) == 1:
    axes = [axes]
fig.suptitle('Predicciones vs Valores Reales (Test Set)', fontweight='bold')

for ax, (name, r) in zip(axes, results.items()):
    y_pred = r['y_pred_test']
    ax.scatter(y_test, y_pred, alpha=0.35, s=15, color='steelblue')
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfecta')
    ax.set_xlabel('Real'); ax.set_ylabel('Predicho')
    ax.set_title(f'{name}\nR²={r["R2_test"]:.3f} MAE={r["MAE_test"]:.3f}')
    ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_pred_vs_real.png', dpi=120, bbox_inches='tight')
plt.close()

# Residuos del mejor modelo
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Análisis de Residuos — {best_name}', fontweight='bold')
residuos = y_test.values - best['y_pred_test']

axes[0].scatter(best['y_pred_test'], residuos, alpha=0.35, s=15, color='steelblue')
axes[0].axhline(0, color='red', ls='--')
axes[0].set_xlabel('Predicho'); axes[0].set_ylabel('Residuo')
axes[0].set_title('Residuos vs Predicho')

axes[1].hist(residuos, bins=40, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='red', ls='--')
axes[1].set_xlabel('Residuo'); axes[1].set_title('Distribución de Residuos')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_residuos.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"\n  Figuras: 07_pred_vs_real.png, 07_residuos.png")
print(f"\n[FASE 7] OK — Evaluacion completa.")

# ─────────────────────────────────────────────────────────────
# FASE 8 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("FASE 8 — FEATURE IMPORTANCE")
print(SEPARATOR)

best_model = best['model']
importances = best_model.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\nRanking de importancia ({best_name}):")
print(f"{'#':>3s} {'Feature':35s} {'Importancia':>12s} {'Acum%':>8s}")
print("-" * 65)
acum = 0
for i, (_, row) in enumerate(feat_imp.iterrows()):
    acum += row['importance']
    print(f"{i+1:3d} {row['feature']:35s} {row['importance']:12.4f} {acum*100:8.1f}%")

# Plot
fig, ax = plt.subplots(figsize=(9, 8))
colors = ['#1a6fa8' if i < 5 else '#5ba3d0' if i < 10 else '#aacde8'
          for i in range(len(feat_imp))]
bars = ax.barh(range(len(feat_imp)), feat_imp['importance'], color=colors, edgecolor='white')
ax.set_yticks(range(len(feat_imp)))
ax.set_yticklabels(feat_imp['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Importancia (Mean Decrease Impurity)')
ax.set_title(f'Feature Importance — {best_name}', fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, feat_imp['importance'])):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figura: 08_feature_importance.png")

print(f"\n[INTERPRETACION ECOLOGICA]")
top3 = feat_imp.head(3)
for _, row in top3.iterrows():
    f = row['feature']
    if 'altura_actual' in f:
        print(f"  {f}: La altura actual es el predictor dominante — arboles mas altos tienen mayor crecimiento absoluto (efecto alometrico)")
    elif 'copa_actual' in f or 'copa_anterior' in f:
        print(f"  {f}: La copa refleja acceso a luz — mayor dosel = mayor capacidad fotosintética = mayor crecimiento")
    elif 'elevacion' in f:
        print(f"  {f}: La elevacion modula temperatura y humedad — afecta directamente la tasa de crecimiento")
    elif 'especie' in f or 'familia' in f:
        print(f"  {f}: La especie determina el nicho ecológico y velocidad de crecimiento intrínseca")
    elif 'gremio' in f:
        print(f"  {f}: El gremio ecológico (pionero/intermedio/tardío) define la estrategia de crecimiento")
    else:
        print(f"  {f}: importancia={row['importance']:.4f}")

print(f"\n[FASE 8] OK — Feature importance completada.")
print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
import os
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")

"""
Script para entrenar y exportar modelos .pkl para la app web.
Ejecutar UNA sola vez desde la carpeta datasets curados.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Crear carpeta destino para los modelos
MODELS_DIR = r'..\agrosense-web\backend\ml\models'
os.makedirs(MODELS_DIR, exist_ok=True)

print("Cargando dataset...")
df = pd.read_excel('dataset_ml.xlsx')

# Feature engineering
df['altura_actual'] = df['altura_anterior'] + df['crecimiento_altura']
df['copa_actual']   = df['copa_anterior']   + df['crecimiento_copa']

# Target
y = df['crecimiento_altura'].copy()

# Features
EXCLUDED = ['unique_tree_id', 'crecimiento_altura', 'crecimiento_copa',
            'monitoreo_label', 'id_parcela']
feature_cols = [c for c in df.columns if c not in EXCLUDED]
X = df[feature_cols].copy()

# Imputar nulls
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Codificar categóricas
cat_cols = X.select_dtypes(include='object').columns.tolist()
le_dict = {}
X_model = X.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_model[col] = le.fit_transform(X_model[col].astype(str))
    le_dict[col] = le

feature_names = X_model.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.20, random_state=42)

print("Entrenando GradientBoostingRegressor...")
gb = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=3, subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
gb_metrics = {
    'r2':   round(r2_score(y_test, y_pred), 4),
    'mae':  round(mean_absolute_error(y_test, y_pred), 4),
    'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
}
print(f"  GB — R2={gb_metrics['r2']}, MAE={gb_metrics['mae']}, RMSE={gb_metrics['rmse']}")

print("Entrenando RandomForestRegressor...")
rf = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=3,
    max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_metrics = {
    'r2':   round(r2_score(y_test, y_pred_rf), 4),
    'mae':  round(mean_absolute_error(y_test, y_pred_rf), 4),
    'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 4)
}
print(f"  RF — R2={rf_metrics['r2']}, MAE={rf_metrics['mae']}, RMSE={rf_metrics['rmse']}")

print("Entrenando IsolationForest...")
num_cols_iso = ['altura_anterior', 'altura_actual', 'copa_anterior', 'copa_actual',
                'crecimiento_altura', 'crecimiento_copa', 'elevacion_m', 'meses_desde_inicio']
num_cols_iso = [c for c in num_cols_iso if c in df.columns]
X_iso = df[num_cols_iso].copy()
for col in X_iso.columns:
    if X_iso[col].isnull().sum() > 0:
        X_iso[col] = X_iso[col].fillna(X_iso[col].median())
scaler = StandardScaler()
X_iso_scaled = scaler.fit_transform(X_iso)
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_iso_scaled)
print(f"  IsolationForest entrenado OK")

# Feature importance (GB)
fi = pd.DataFrame({
    'feature': feature_names,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

# Guardar todo en un bundle
bundle = {
    'gb_model':        gb,
    'rf_model':        rf,
    'iso_model':       iso,
    'iso_scaler':      scaler,
    'iso_cols':        num_cols_iso,
    'label_encoders':  le_dict,
    'feature_names':   feature_names,
    'cat_cols':        cat_cols,
    'feature_importance': fi.to_dict(orient='records'),
    'metrics': {
        'GradientBoosting': gb_metrics,
        'RandomForest':     rf_metrics,
    },
    'dataset_stats': {
        'n_rows':    int(len(df)),
        'n_species': int(df['especie'].nunique()),
        'n_trees':   int(df['unique_tree_id'].nunique()),
        'monitoreos': sorted(df['monitoreo'].unique().tolist()),
    }
}

bundle_path = os.path.join(MODELS_DIR, 'model_bundle.pkl')
with open(bundle_path, 'wb') as f:
    pickle.dump(bundle, f)

print(f"\nBundle guardado en: {bundle_path}")
print(f"  Tamanio: {os.path.getsize(bundle_path)/1024/1024:.1f} MB")
print("\nExportacion completa. Todos los modelos listos.")

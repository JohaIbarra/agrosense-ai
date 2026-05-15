import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel(r'dataset_ml.xlsx')

print("=" * 60)
print("VERIFICACION DE COLUMNAS OBJETIVO")
print("=" * 60)

target_cols = ['altura_actual', 'copa_actual']
for col in target_cols:
    if col in df.columns:
        print(f"  [SI] '{col}' EXISTE")
    else:
        print(f"  [NO] '{col}' NO EXISTE en el dataset")

print("\n" + "=" * 60)
print("COLUMNAS RELACIONADAS CON ALTURA Y COPA")
print("=" * 60)

keywords = ['altura', 'copa', 'crecimiento']
related = [c for c in df.columns if any(k in c.lower() for k in keywords)]
for col in related:
    print(f"\n--- {col} ---")
    print(f"  dtype: {df[col].dtype}")
    print(f"  nulls: {df[col].isnull().sum()}/{len(df)}")
    print(f"  min: {df[col].min()}")
    print(f"  max: {df[col].max()}")
    print(f"  mean: {df[col].mean():.4f}")
    print(f"  Primeros 10: {df[col].head(10).tolist()}")

print("\n" + "=" * 60)
print("VERIFICACION MATEMATICA DE RECONSTRUCCION")
print("=" * 60)

# altura_actual = altura_anterior + crecimiento_altura
if 'altura_anterior' in df.columns and 'crecimiento_altura' in df.columns:
    altura_calc = df['altura_anterior'] + df['crecimiento_altura']
    print("\naltura_actual = altura_anterior + crecimiento_altura")
    print(f"  Posible: SI")
    print(f"  Nulls en altura_anterior: {df['altura_anterior'].isnull().sum()}")
    print(f"  Nulls en crecimiento_altura: {df['crecimiento_altura'].isnull().sum()}")
    print(f"  Nulls en resultado: {altura_calc.isnull().sum()}")
    print(f"  Resultado - min: {altura_calc.min():.4f}, max: {altura_calc.max():.4f}, mean: {altura_calc.mean():.4f}")
    print(f"  Primeros 10 valores calculados: {altura_calc.head(10).tolist()}")
    print(f"  Valores negativos: {(altura_calc < 0).sum()}")
else:
    print("  No es posible reconstruir altura_actual - faltan columnas")

# copa_actual = copa_anterior + crecimiento_copa
if 'copa_anterior' in df.columns and 'crecimiento_copa' in df.columns:
    copa_calc = df['copa_anterior'] + df['crecimiento_copa']
    print("\ncopa_actual = copa_anterior + crecimiento_copa")
    print(f"  Posible: SI")
    print(f"  Nulls en copa_anterior: {df['copa_anterior'].isnull().sum()}")
    print(f"  Nulls en crecimiento_copa: {df['crecimiento_copa'].isnull().sum()}")
    print(f"  Nulls en resultado: {copa_calc.isnull().sum()}")
    print(f"  Resultado - min: {copa_calc.min():.4f}, max: {copa_calc.max():.4f}, mean: {copa_calc.mean():.4f}")
    print(f"  Primeros 10 valores calculados: {copa_calc.head(10).tolist()}")
    print(f"  Valores negativos: {(copa_calc < 0).sum()}")
else:
    print("  No es posible reconstruir copa_actual - faltan columnas")

print("\n" + "=" * 60)
print("VALIDACION CRUZADA - Comparar monitoreos sucesivos")
print("=" * 60)
# Si tree X en monitoreo N tiene altura_anterior = A
# Y tree X en monitoreo N-1 tiene altura_anterior + crecimiento = B
# Deberia ser A ~= B (la altura_actual del monitoreo anterior deberia ser la altura_anterior del siguiente)

if 'unique_tree_id' in df.columns and 'monitoreo' in df.columns:
    altura_calc = df['altura_anterior'] + df['crecimiento_altura']
    df_check = df[['unique_tree_id', 'monitoreo', 'altura_anterior', 'crecimiento_altura']].copy()
    df_check['altura_calculada'] = altura_calc
    
    # Unir monitoreo N con monitoreo N+1 del mismo arbol
    df_n = df_check.rename(columns={'monitoreo': 'mon', 'altura_anterior': 'alt_ant_n', 'crecimiento_altura': 'crec_n', 'altura_calculada': 'alt_calc_n'})
    df_n1 = df_check[['unique_tree_id', 'monitoreo', 'altura_anterior']].rename(columns={'monitoreo': 'mon_next', 'altura_anterior': 'alt_ant_next'})
    df_n['mon_next'] = df_n['mon'] + 1
    
    merged = df_n.merge(df_n1, on=['unique_tree_id', 'mon_next'], how='inner')
    
    if len(merged) > 0:
        merged['diff'] = abs(merged['alt_calc_n'] - merged['alt_ant_next'])
        print(f"  Pares monitoreo consecutivo encontrados: {len(merged)}")
        print(f"  Diferencia promedio (altura_calc_n vs altura_ant_n+1): {merged['diff'].mean():.6f}")
        print(f"  Diferencia max: {merged['diff'].max():.6f}")
        print(f"  Pares con diferencia = 0 (match exacto): {(merged['diff'] == 0).sum()}/{len(merged)}")
        print(f"  Pares con diferencia < 0.01: {(merged['diff'] < 0.01).sum()}/{len(merged)}")
        print(f"\n  Muestra de 5 pares:")
        sample = merged.head(5)
        for _, row in sample.iterrows():
            print(f"    Tree={row['unique_tree_id']}, Mon {int(row['mon'])}: alt_ant={row['alt_ant_n']:.3f} + crec={row['crec_n']:.3f} = {row['alt_calc_n']:.3f} | Mon {int(row['mon_next'])}: alt_ant={row['alt_ant_next']:.3f}")
    else:
        print("  No se encontraron pares consecutivos para validar")

# Mismo para copa
if 'copa_anterior' in df.columns and 'crecimiento_copa' in df.columns:
    copa_calc_vals = df['copa_anterior'] + df['crecimiento_copa']
    df_check2 = df[['unique_tree_id', 'monitoreo', 'copa_anterior', 'crecimiento_copa']].copy()
    df_check2['copa_calculada'] = copa_calc_vals
    
    df_cn = df_check2.rename(columns={'monitoreo': 'mon', 'copa_anterior': 'copa_ant_n', 'crecimiento_copa': 'crec_copa_n', 'copa_calculada': 'copa_calc_n'})
    df_cn1 = df_check2[['unique_tree_id', 'monitoreo', 'copa_anterior']].rename(columns={'monitoreo': 'mon_next', 'copa_anterior': 'copa_ant_next'})
    df_cn['mon_next'] = df_cn['mon'] + 1
    
    merged_c = df_cn.merge(df_cn1, on=['unique_tree_id', 'mon_next'], how='inner')
    
    if len(merged_c) > 0:
        merged_c['diff_copa'] = abs(merged_c['copa_calc_n'] - merged_c['copa_ant_next'])
        print(f"\n  COPA - Pares consecutivos: {len(merged_c)}")
        print(f"  Diferencia promedio (copa_calc_n vs copa_ant_n+1): {merged_c['diff_copa'].mean():.6f}")
        print(f"  Match exacto: {(merged_c['diff_copa'] == 0).sum()}/{len(merged_c)}")
        print(f"  Diferencia < 0.01: {(merged_c['diff_copa'] < 0.01).sum()}/{len(merged_c)}")

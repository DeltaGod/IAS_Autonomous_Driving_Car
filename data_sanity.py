import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_benchmark(train_csv="dataset_train.csv", val_csv="dataset_val.csv"):
    missing = [p for p in (train_csv, val_csv) if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] No se encontraron los archivos: {', '.join(missing)}.")
        print("Corré primero 'build_global_csv.py' para generarlos.")
        return

    print("Cargando datasets (train + val)...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_train["split"] = "train"
    df_val["split"] = "val"
    df = pd.concat([df_train, df_val], ignore_index=True)

    print("\n============================================================")
    print("📊 REPORTE DE DATA SANITY Y BENCHMARKING")
    print("============================================================")
    print(f"Split por sesión -> Train: {len(df_train)} frames | Val: {len(df_val)} frames")

    # 0. DISTRIBUCIÓN DE CLASES POR SPLIT
    print("\n--- DISTRIBUCIÓN DE CLASES POR SPLIT (%) ---")
    by_split = (
        df.groupby("split")["behavior"].value_counts(normalize=True).mul(100).round(1)
        .unstack(fill_value=0)
    )
    print(by_split.to_string())

    # 1. VERIFICACIÓN DE INTEGRIDAD BÁSICA
    total_rows = len(df)
    null_counts = df.isnull().sum().sum()
    print(f"Total de registros: {total_rows}")
    print(f"Valores nulos en el CSV: {null_counts} " + ("(¡PELIGRO!)" if null_counts > 0 else "(Ok)"))

    # 2. VERIFICACIÓN DE IMÁGENES EN DISCO
    print("\nVerificando existencia física de las imágenes...")
    # Tomamos una muestra aleatoria para no saturar el disco si son 20k+ imágenes
    sample_size = min(2000, total_rows)
    missing_images = 0
    for path in df['image_path'].sample(sample_size):
        if not os.path.exists(path):
            missing_images += 1
    
    missing_ratio = missing_images / sample_size
    print(f"Imágenes perdidas en disco (Muestra de {sample_size}): {missing_images}")
    if missing_ratio > 0:
        print(f"[ALERTA ROJA] Aproximadamente el {missing_ratio*100:.1f}% de las rutas están rotas.")
    else:
        print("Integridad de rutas de imagen: (Ok)")

    # 3. RATIO DE FUENTES (REAL VS INTERPOLADO)
    print("\n--- DISTRIBUCIÓN DE FUENTE DE COMANDOS ---")
    source_dist = df['source'].value_counts(normalize=True) * 100
    print(source_dist.round(2).astype(str) + " %")
    if 'interp' in source_dist and source_dist['interp'] > 85:
        print("[ADVERTENCIA] Tasa de interpolación extremadamente alta. Cuidado con el overfitting al algoritmo.")

    # 4. BALANCE DE COMPORTAMIENTO (BEHAVIOR)
    print("\n--- DISTRIBUCIÓN DE CLASES (BEHAVIOR) ---")
    behavior_dist = df['behavior'].value_counts(normalize=True) * 100
    print(behavior_dist.round(2).astype(str) + " %")

    # 5. ANÁLISIS DE DELTA T (TIEMPO ENTRE FRAMES)
    print("\n--- ANÁLISIS DE LATENCIA DE CÁMARA (Delta T) ---")
    df = df.sort_values(by=['record', 'time_in_ms'])
    df['delta_t'] = df.groupby('record')['time_in_ms'].diff()
    
    dt_mean = df['delta_t'].mean()
    dt_median = df['delta_t'].median()
    dt_max = df['delta_t'].max()
    fps_aprox = 1000 / dt_median if dt_median > 0 else 0
    
    print(f"Delta T Medio:   {dt_mean:.2f} ms")
    print(f"Delta T Mediana: {dt_median:.2f} ms (Aprox {fps_aprox:.1f} FPS)")
    print(f"Delta T Máximo:  {dt_max:.2f} ms")
    
    saltos_criticos = len(df[df['delta_t'] > 1000])
    print(f"Frames con delay > 1 segundo: {saltos_criticos}")

    # 6. ESTADÍSTICAS DE PWM (SPEED A y B)
    print("\n--- ESTADÍSTICAS DE MOTORES ---")
    print("Velocidad Motor Izq (A):")
    print(df['speedA'].describe()[['mean', 'std', 'min', '50%', 'max']].to_string())
    print("Velocidad Motor Der (B):")
    print(df['speedB'].describe()[['mean', 'std', 'min', '50%', 'max']].to_string())

    # ==========================================
    # GENERACIÓN DE GRÁFICOS VISUALES
    # ==========================================
    print("\nGenerando gráficos de diagnóstico ('benchmark_report.png')...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Benchmark del Dataset de Conducción Autónoma', fontsize=16)

    # Gráfico 1: Behavior Counts
    sns.countplot(data=df, x='behavior', order=df['behavior'].value_counts().index, ax=axes[0, 0], hue='source')
    axes[0, 0].set_title('Distribución de Clases por Fuente')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Gráfico 2: Delta T Distribution
    sns.histplot(data=df[df['delta_t'] < 2000], x='delta_t', bins=50, ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Distribución de Latencia entre Frames (< 2s)')
    axes[0, 1].set_xlabel('Delta T (ms)')

    # Gráfico 3: Speed distributions
    sns.kdeplot(data=df, x='speedA', label='Motor Izq (A)', fill=True, ax=axes[1, 0])
    sns.kdeplot(data=df, x='speedB', label='Motor Der (B)', fill=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribución Densidad de Velocidades PWM')
    axes[1, 0].legend()

    # Gráfico 4: Evolución de velocidades en una sesión aleatoria
    muestra_record = df['record'].unique()[0]
    df_sample = df[df['record'] == muestra_record].reset_index()
    # Tomamos solo 200 frames para que se vea claro
    df_sample = df_sample.head(200)
    sns.lineplot(data=df_sample, x=df_sample.index, y='speedA', label='Motor A', ax=axes[1, 1])
    sns.lineplot(data=df_sample, x=df_sample.index, y='speedB', label='Motor B', ax=axes[1, 1])
    axes[1, 1].set_title(f'Rastro de Velocidad (Sample: {muestra_record})')
    axes[1, 1].set_xlabel('Nº de Frame')

    plt.tight_layout()
    plt.savefig('benchmark_report.png', dpi=300)
    print("¡Listo! Análisis completo.")

if __name__ == "__main__":
    run_benchmark()
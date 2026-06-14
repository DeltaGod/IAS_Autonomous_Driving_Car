import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from config import CFG

def run_benchmark(train_csv=None, val_csv=None):
    train_csv = train_csv or CFG.train_csv
    val_csv = val_csv or CFG.val_csv
    missing = [p for p in (train_csv, val_csv) if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] No se encontraron los archivos: {', '.join(missing)}.")
        print("Corre primero 'build_global_csv.py' para generarlos.")
        return

    print("Cargando datasets (train + val)...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_train["split"] = "train"
    df_val["split"] = "val"
    df = pd.concat([df_train, df_val], ignore_index=True)

    print("\n============================================================")
    print("[graf] REPORTE DE DATA SANITY Y BENCHMARKING")
    print("============================================================")
    print(f"Split por sesion -> Train: {len(df_train)} frames | Val: {len(df_val)} frames")

    # 0. DISTRIBUCION DE CLASES POR SPLIT
    print("\n--- DISTRIBUCION BEHAVIOR GLOBAL POR SPLIT (%) ---")
    by_split = (
        df.groupby("split")["behavior"].value_counts(normalize=True).mul(100).round(1)
        .unstack(fill_value=0)
    )
    print(by_split.to_string())

    # 0b. DISTRIBUCION POR MOTOR (lo que realmente predice el modelo)
    print("\n--- DISTRIBUCION DIRECCION POR MOTOR Y SPLIT (%) ---")
    for split in ["train", "val"]:
        sub = df[df["split"] == split]
        per_motor = pd.DataFrame({
            "motorA": sub["behaviorA"].value_counts(normalize=True) * 100,
            "motorB": sub["behaviorB"].value_counts(normalize=True) * 100,
        }).round(1).fillna(0)
        print(f"[{split}]")
        print(per_motor.to_string())

    # 1. VERIFICACION DE INTEGRIDAD BASICA
    total_rows = len(df)
    null_counts = df.isnull().sum().sum()
    print(f"Total de registros: {total_rows}")
    print(f"Valores nulos en el CSV: {null_counts} " + ("(PELIGRO!)" if null_counts > 0 else "(Ok)"))

    # 2. VERIFICACION DE IMAGENES EN DISCO
    print("\nVerificando existencia fisica de las imagenes...")
    # Tomamos una muestra aleatoria para no saturar el disco si son 20k+ imagenes
    sample_size = min(2000, total_rows)
    missing_images = 0
    for path in df['image_path'].sample(sample_size):
        if not os.path.exists(path):
            missing_images += 1
    
    missing_ratio = missing_images / sample_size
    print(f"Imagenes perdidas en disco (Muestra de {sample_size}): {missing_images}")
    if missing_ratio > 0:
        print(f"[ALERTA ROJA] Aproximadamente el {missing_ratio*100:.1f}% de las rutas estan rotas.")
    else:
        print("Integridad de rutas de imagen: (Ok)")

    # 3. RATIO DE FUENTES (REAL VS INTERPOLADO)
    print("\n--- DISTRIBUCION DE FUENTE DE COMANDOS ---")
    source_dist = df['source'].value_counts(normalize=True) * 100
    print(source_dist.round(2).astype(str) + " %")
    if 'interp' in source_dist and source_dist['interp'] > 85:
        print("[ADVERTENCIA] Tasa de interpolacion extremadamente alta. Cuidado con el overfitting al algoritmo.")

    # 4. BALANCE DE COMPORTAMIENTO (BEHAVIOR)
    print("\n--- DISTRIBUCION DE CLASES (BEHAVIOR) ---")
    behavior_dist = df['behavior'].value_counts(normalize=True) * 100
    print(behavior_dist.round(2).astype(str) + " %")

    # 5. ANALISIS DE DELTA T (TIEMPO ENTRE FRAMES)
    print("\n--- ANALISIS DE LATENCIA DE CAMARA (Delta T) ---")
    df = df.sort_values(by=['record', 'time_in_ms'])
    df['delta_t'] = df.groupby('record')['time_in_ms'].diff()
    
    dt_mean = df['delta_t'].mean()
    dt_median = df['delta_t'].median()
    dt_max = df['delta_t'].max()
    fps_aprox = 1000 / dt_median if dt_median > 0 else 0
    
    print(f"Delta T Medio:   {dt_mean:.2f} ms")
    print(f"Delta T Mediana: {dt_median:.2f} ms (Aprox {fps_aprox:.1f} FPS)")
    print(f"Delta T Maximo:  {dt_max:.2f} ms")
    
    saltos_criticos = len(df[df['delta_t'] > 1000])
    print(f"Frames con delay > 1 segundo: {saltos_criticos}")

    # 6. ESTADISTICAS DE PWM (SPEED A y B)
    print("\n--- ESTADISTICAS DE MOTORES ---")
    print("Velocidad Motor Izq (A):")
    print(df['speedA'].describe()[['mean', 'std', 'min', '50%', 'max']].to_string())
    print("Velocidad Motor Der (B):")
    print(df['speedB'].describe()[['mean', 'std', 'min', '50%', 'max']].to_string())

    # ==========================================
    # GENERACION DE GRAFICOS VISUALES
    # ==========================================
    print("\nGenerando graficos de diagnostico ('benchmark_report.png')...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('Benchmark del Dataset de Conduccion Autonoma', fontsize=16)

    dir_order = ["STOP", "FORWARD", "BACKWARD"]

    def motor_pct(col):
        """% de cada direccion DENTRO de cada split (para que val no quede aplastado)."""
        return (df.groupby('split')[col].value_counts(normalize=True).mul(100)
                  .rename('pct').reset_index())

    # Grafico 1: Behavior global por fuente (real/interp)
    sns.countplot(data=df, x='behavior', order=df['behavior'].value_counts().index,
                  ax=axes[0, 0], hue='source')
    axes[0, 0].set_title('Behavior GLOBAL (5 clases) por Fuente')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Grafico 2: Direccion Motor A (IZQ) por split
    sns.barplot(data=motor_pct('behaviorA'), x='behaviorA', y='pct', hue='split',
                order=dir_order, ax=axes[0, 1])
    axes[0, 1].set_title('Direccion Motor A (IZQ) por Split')
    axes[0, 1].set_xlabel('direccion'); axes[0, 1].set_ylabel('% dentro del split')

    # Grafico 3: Direccion Motor B (DER) por split
    sns.barplot(data=motor_pct('behaviorB'), x='behaviorB', y='pct', hue='split',
                order=dir_order, ax=axes[0, 2])
    axes[0, 2].set_title('Direccion Motor B (DER) por Split')
    axes[0, 2].set_xlabel('direccion'); axes[0, 2].set_ylabel('% dentro del split')

    # Grafico 4: Delta T
    sns.histplot(data=df[df['delta_t'] < 2000], x='delta_t', bins=50, ax=axes[1, 0], color='coral')
    axes[1, 0].set_title('Distribucion de Latencia entre Frames (< 2s)')
    axes[1, 0].set_xlabel('Delta T (ms)')

    # Grafico 5: Densidad de velocidades PWM
    sns.kdeplot(data=df, x='speedA', label='Motor Izq (A)', fill=True, ax=axes[1, 1])
    sns.kdeplot(data=df, x='speedB', label='Motor Der (B)', fill=True, ax=axes[1, 1])
    axes[1, 1].set_title('Densidad de Velocidades PWM')
    axes[1, 1].legend()

    # Grafico 6: Rastro de velocidades en la sesion con MAS frames (evita chunks seq de 1 fila)
    muestra_record = df['record'].value_counts().idxmax()
    df_sample = df[df['record'] == muestra_record].sort_values('time_in_ms').reset_index().head(200)
    sns.lineplot(data=df_sample, x=df_sample.index, y='speedA', label='Motor A', ax=axes[1, 2])
    sns.lineplot(data=df_sample, x=df_sample.index, y='speedB', label='Motor B', ax=axes[1, 2])
    axes[1, 2].set_title(f'Rastro de Velocidad (Sample: {muestra_record})')
    axes[1, 2].set_xlabel('No de Frame')

    plt.tight_layout()
    plt.savefig('benchmark_report.png', dpi=300)
    print("Listo! Analisis completo.")

if __name__ == "__main__":
    run_benchmark()
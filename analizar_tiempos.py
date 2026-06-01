import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar el estilo de los gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Ruta base del dataset (relativa al script)
dataset_path = Path(__file__).parent / "DataSet" / "Records (1)"

# Obtener todas las carpetas de registros
record_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith("#Record")])

print(f"Se encontraron {len(record_folders)} registros\n")

all_time_diffs = []
results_by_record = {}

# Procesar cada registro
for folder in record_folders:
    csv_file = folder / "labels.csv"
    
    if csv_file.exists():
        print(f"Procesando: {folder.name}")
        
        # Leer el CSV
        df = pd.read_csv(csv_file, sep=";", nrows=10)  # Primeras 10 filas como mencionaste
        
        # Calcular diferencias de tiempo entre registros consecutivos
        times = df['time_in_ms'].values
        time_diffs = np.diff(times)
        
        # Estadísticas del registro actual
        metrics = {
            'promedio_ms': np.mean(time_diffs),
            'desv_estandar_ms': np.std(time_diffs),
            'min_ms': np.min(time_diffs),
            'max_ms': np.max(time_diffs),
            'mediana_ms': np.median(time_diffs),
            'q25_ms': np.percentile(time_diffs, 25),
            'q75_ms': np.percentile(time_diffs, 75),
            'cantidad_registros': len(time_diffs)
        }
        
        results_by_record[folder.name] = metrics
        all_time_diffs.extend(time_diffs)
        
        print(f"  Promedio: {metrics['promedio_ms']:.2f} ms")
        print(f"  Desv. Est.: {metrics['desv_estandar_ms']:.2f} ms")
        print(f"  Rango: [{metrics['min_ms']:.2f}, {metrics['max_ms']:.2f}] ms")
        print(f"  Mediana: {metrics['mediana_ms']:.2f} ms\n")

# Estadísticas globales
print("="*60)
print("ESTADÍSTICAS GLOBALES (Todos los registros)")
print("="*60)

all_diffs_array = np.array(all_time_diffs)
print(f"Promedio global: {np.mean(all_diffs_array):.2f} ms")
print(f"Desv. Est. global: {np.std(all_diffs_array):.2f} ms")
print(f"Mínimo: {np.min(all_diffs_array):.2f} ms")
print(f"Máximo: {np.max(all_diffs_array):.2f} ms")
print(f"Mediana: {np.median(all_diffs_array):.2f} ms")
print(f"Q25: {np.percentile(all_diffs_array, 25):.2f} ms")
print(f"Q75: {np.percentile(all_diffs_array, 75):.2f} ms")
print(f"Total de diferencias: {len(all_diffs_array)}\n")

# Crear visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histograma
axes[0, 0].hist(all_diffs_array, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(all_diffs_array), color='red', linestyle='--', linewidth=2, label=f"Media: {np.mean(all_diffs_array):.2f} ms")
axes[0, 0].axvline(np.median(all_diffs_array), color='green', linestyle='--', linewidth=2, label=f"Mediana: {np.median(all_diffs_array):.2f} ms")
axes[0, 0].set_xlabel('Diferencia de tiempo (ms)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Diferencias de Tiempo')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(all_diffs_array, vert=True)
axes[0, 1].set_ylabel('Diferencia de tiempo (ms)')
axes[0, 1].set_title('Box Plot de Diferencias de Tiempo')
axes[0, 1].grid(True, alpha=0.3)

# Gráfico de densidad
axes[1, 0].hist(all_diffs_array, bins=50, density=True, color='steelblue', alpha=0.6, edgecolor='black')
from scipy import stats
mu, sigma = np.mean(all_diffs_array), np.std(all_diffs_array)
x = np.linspace(all_diffs_array.min(), all_diffs_array.max(), 100)
axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Distribución Normal')
axes[1, 0].set_xlabel('Diferencia de tiempo (ms)')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].set_title('Densidad de Probabilidad')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Estadísticas por registro
record_names = [name.replace("#Record_", "") for name in results_by_record.keys()]
promedios = [results_by_record[name]['promedio_ms'] for name in results_by_record.keys()]
desv_estandar = [results_by_record[name]['desv_estandar_ms'] for name in results_by_record.keys()]

x_pos = np.arange(len(record_names))
axes[1, 1].bar(x_pos, promedios, yerr=desv_estandar, capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Registro')
axes[1, 1].set_ylabel('Promedio ± Desv. Est. (ms)')
axes[1, 1].set_title('Promedio de Diferencias por Registro')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(record_names, rotation=45, ha='right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# Guardar gráfico en la carpeta del proyecto
output_path = Path(__file__).parent / "analisis_tiempos.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {output_path}")
plt.show()

# Tabla resumen
print("\n" + "="*60)
print("TABLA RESUMEN POR REGISTRO")
print("="*60)
df_resumen = pd.DataFrame(results_by_record).T
print(df_resumen.to_string())
csv_path = Path(__file__).parent / "resumen_tiempos.csv"
df_resumen.to_csv(csv_path)
print(f"\nTabla guardada en: {csv_path}")

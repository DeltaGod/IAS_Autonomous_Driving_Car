import wandb
import pandas as pd
import os

def get_latest_local_run_id():
    """Busca dinámicamente el ID de la última corrida leyendo el symlink de W&B."""
    # Resolución dinámica del directorio base (donde se encuentra este script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wandb_dir = os.path.join(base_dir, "wandb")
    latest_run_link = os.path.join(wandb_dir, "latest-run")
    
    if not os.path.exists(latest_run_link):
        print(f"[ERROR] No se encontró el directorio {latest_run_link}.")
        print("Asegúrate de haber ejecutado un entrenamiento en este directorio.")
        return None

    # Leer hacia dónde apunta el enlace simbólico
    real_path = os.path.realpath(latest_run_link)
    folder_name = os.path.basename(real_path)  # ej. run-20260608_121155-8sv1zi1v
    
    # El ID es la cadena alfanumérica después del último guion
    run_id = folder_name.split('-')[-1]
    return run_id

def download_run_history():
    run_id = get_latest_local_run_id()
    if not run_id:
        return

    print("Conectando con la API de Weights & Biases...")
    api = wandb.Api()

    # ==========================================
    # SETEA TU USUARIO AQUÍ UNA SOLA VEZ
    # ==========================================
    USERNAME = "juancbaudino-cole-nationale-d-ing-nieurs-de-brest" 
    PROJECT_NAME = "Autonomous-Driving-ENIB"
    
    run_path = f"{USERNAME}/{PROJECT_NAME}/{run_id}"
    
    try:
        run = api.run(run_path)
        print(f"Descargando historial de la sesión local más reciente: {run.name} (ID: {run_id})...")
        
        # Extraer métricas históricas
        history = run.history()
        
        # Limpieza de columnas internas del sistema
        cols_to_keep = [col for col in history.columns if not col.startswith('_')]
        clean_history = history[cols_to_keep]
        
        # Guardado dinámico en la raíz del proyecto
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_filename = f"wandb_history_{run_id}.csv"
        output_path = os.path.join(base_dir, output_filename)
        
        clean_history.to_csv(output_path, index=False)
        
        print("\n============================================================")
        print(f"✅ ¡Descarga completa! Archivo guardado en:")
        print(f"   {output_path}")
        print("============================================================")
        
    except wandb.errors.CommError:
        print("\n[ERROR DE CONEXIÓN] No se pudo acceder a la corrida.")
        print(f"Verifica que tu USERNAME ('{USERNAME}') sea correcto y tengas permisos.")
    except Exception as e:
        print(f"\n[ERROR INESPERADO] {e}")

if __name__ == "__main__":
    download_run_history()
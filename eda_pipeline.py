import pandas as pd
import glob
import os

def decode_action(row):
    # Direcciones base según GPIO
    left_fwd = (row['GPIO1'] == 1) and (row['GPIO2'] == 0)
    left_bwd = (row['GPIO1'] == 0) and (row['GPIO2'] == 1)
    right_fwd = (row['GPIO3'] == 0) and (row['GPIO4'] == 1)
    right_bwd = (row['GPIO3'] == 1) and (row['GPIO4'] == 0)
    
    # Velocidades
    v_left = row['speedA']
    v_right = row['speedB']
    
    # Lógica de decisión
    if v_left == 0 and v_right == 0:
        return 'Freno_Total'
        
    elif left_fwd and right_fwd:
        # Ambos hacia adelante, definimos el giro por la diferencia de velocidades
        if v_left > v_right:
            return 'Giro_Derecha_Avanzando'
        elif v_right > v_left:
            return 'Giro_Izquierda_Avanzando'
        else:
            return 'Adelante_Recto'
            
    elif left_bwd and right_bwd:
        return 'Reversa'
        
    # Giros sobre el propio eje (Tank turn)
    elif left_fwd and right_bwd:
        return 'Giro_Derecha_Eje'
    elif right_fwd and left_bwd:
        return 'Giro_Izquierda_Eje'
        
    else:
        return 'Accion_Compleja/Transicion'

def run_eda():
    # Buscar todos los archivos labels.csv recursivamente
    path = "DataSet/**/*.csv"
    all_files = glob.glob(path, recursive=True)
    
    if not all_files:
        print("No se encontraron archivos .csv. Verifica el path.")
        return
        
    df_list = []
    for file in all_files:
        # Algunos de tus logs usaban punto y coma (;), verifica si es coma o punto y coma
        # Usaremos el separador dinámico de pandas
        df = pd.read_csv(file, sep=None, engine='python')
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Aplicar la decodificación
    full_df['Direccion_Asumida'] = full_df.apply(decode_action, axis=1)
    
    print("============================================================")
    print("📊 REPORTE DEL DATASET (FASE 1)")
    print("============================================================")
    print(f"Total de frames (filas) registrados: {len(full_df)}")
    print("\n--- DISTRIBUCIÓN DE DIRECCIONES ---")
    print(full_df['Direccion_Asumida'].value_counts(normalize=True) * 100)
    
    print("\n--- DISTRIBUCIÓN DE VELOCIDADES (Motor Izquierdo) ---")
    print(full_df['speedA'].value_counts(normalize=True) * 100)
    
    print("\n--- DISTRIBUCIÓN DE VELOCIDADES (Motor Derecho) ---")
    print(full_df['speedB'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    run_eda()
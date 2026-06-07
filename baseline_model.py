import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time

def load_and_flatten_images(df, img_size=(32, 32)):
    """Carga imágenes, las pasa a grises y las aplasta en vectores 1D."""
    X = []
    y = []
    groups = []
    
    print(f"Cargando y procesando {len(df)} imágenes a {img_size}...")
    start = time.time()
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue
            
        # Leer en escala de grises para simplificar al máximo
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, img_size)
        
        # Aplastar la imagen (32x32 = vector de 1024 características)
        X.append(img_resized.flatten())
        y.append(row['behavior'])
        groups.append(row['record']) # El ID del chunk para evitar Data Leakage
        
    print(f"Carga completada en {time.time() - start:.2f} segundos.")
    return np.array(X), np.array(y), np.array(groups)

def run_baseline():
    df = pd.read_csv("dataset_global.csv")
    
    # 1. Cargar datos
    X, y, groups = load_and_flatten_images(df)
    
    # 2. Split riguroso por Grupos (Sesiones)
    # El 20% de las sesiones irán a validación, el 80% a entrenamiento
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"\nDistribución del Split (Cero Data Leakage):")
    print(f"Datos de Entrenamiento: {len(X_train)}")
    print(f"Datos de Validación:  {len(X_val)}")
    
    # 3. Entrenar Random Forest
    print("\nEntrenando Random Forest Contralor (Esto puede tomar unos minutos)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    
    start = time.time()
    rf.fit(X_train, y_train)
    print(f"Entrenamiento completado en {time.time() - start:.2f} segundos.")
    
    # 4. Evaluación
    print("\nEvaluando en el set de validación aislado...")
    y_pred = rf.predict(X_val)
    
    print("\n============================================================")
    print("📊 REPORTE DEL MODELO CONTRALOR (BASELINE)")
    print("============================================================")
    print(classification_report(y_val, y_pred, zero_division=0))

if __name__ == "__main__":
    run_baseline()
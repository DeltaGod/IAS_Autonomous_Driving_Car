"""
baseline_model.py
=================
Baseline NO-deep para contrastar con MotorControlNet. Misma tarea:
IMAGEN -> dirección {STOP,FORWARD,BACKWARD} + velocidad, por motor (A y B).

Random Forest sobre imágenes en grises 32x32 aplanadas:
  - 2 clasificadores de dirección (motor A, motor B), class_weight='balanced'.
  - 2 regresores de velocidad (motor A, motor B), MAE como métrica.
Split por sesión ya materializado: dataset_train.csv / dataset_val.csv.
"""

import pandas as pd
import numpy as np
import cv2
import os
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error


def load_dataset(csv_path, img_size=(32, 32)):
    """Carga imágenes (grises, aplanadas) y los targets por motor."""
    df = pd.read_csv(csv_path)
    X, dirA, dirB, spdA, spdB = [], [], [], [], []
    print(f"Cargando {len(df)} imágenes de {csv_path} a {img_size}...")
    start = time.time()
    for _, row in df.iterrows():
        if not os.path.exists(row['image_path']):
            continue
        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        X.append(cv2.resize(img, img_size).flatten())
        dirA.append(row['behaviorA']); dirB.append(row['behaviorB'])
        spdA.append(float(row['speedA'])); spdB.append(float(row['speedB']))
    print(f"  -> {len(X)} cargadas en {time.time() - start:.1f}s")
    return (np.array(X), np.array(dirA), np.array(dirB),
            np.array(spdA), np.array(spdB))


def run_baseline():
    X_tr, dirA_tr, dirB_tr, spdA_tr, spdB_tr = load_dataset("dataset_train.csv")
    X_va, dirA_va, dirB_va, spdA_va, spdB_va = load_dataset("dataset_val.csv")

    print(f"\nSplit por sesión -> Train: {len(X_tr)} | Val: {len(X_va)}")
    labels = ["STOP", "FORWARD", "BACKWARD"]

    # --- DIRECCIÓN: un Random Forest por motor ---
    for name, y_tr, y_va in [("A (IZQ)", dirA_tr, dirA_va), ("B (DER)", dirB_tr, dirB_va)]:
        print(f"\nEntrenando RF dirección Motor {name}...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)
        print(f"========== DIRECCIÓN MOTOR {name} ==========")
        print(classification_report(y_va, y_pred, labels=labels, zero_division=0))
        print("Matriz de confusión (filas=real, cols=pred), orden", labels)
        print(confusion_matrix(y_va, y_pred, labels=labels))

    # --- VELOCIDAD: un regresor por motor ---
    for name, y_tr, y_va in [("A (IZQ)", spdA_tr, spdA_va), ("B (DER)", spdB_tr, spdB_va)]:
        print(f"\nEntrenando RF velocidad Motor {name}...")
        reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        reg.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_va, reg.predict(X_va))
        print(f"MAE velocidad Motor {name}: {mae:.2f} (escala 0-100)")


if __name__ == "__main__":
    run_baseline()

"""
model_00_baseline.py — Modelo 0: BASELINE no-deep (Random Forest).
===================================================================
El PISO contra el que se contrastan los modelos deep. Misma tarea:
IMAGEN -> dirección {STOP,FORWARD,BACKWARD} + velocidad, por motor (A y B).

Random Forest sobre imágenes en grises 32x32 aplanadas:
  - 2 clasificadores de dirección (motor A, motor B), class_weight='balanced'.
  - 2 regresores de velocidad (motor A, motor B), MAE como métrica.
Split por sesión ya materializado: dataset_train.csv / dataset_val.csv.

Hiperparámetros CONGELADOS abajo (esto ES el modelo). NO usa Lightning ni GPU.
Correr:  .venv/bin/python model_00_baseline.py  ->  resultados en results/model_00_baseline/
"""
import os
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

# ===== hiperparámetros CONGELADOS de este modelo =====
TRAIN_CSV = "dataset_train.csv"
VAL_CSV = "dataset_val.csv"
IMG_SIZE = (32, 32)        # imágenes en grises, aplanadas a 1024 features
N_ESTIMATORS = 100
MAX_DEPTH = 15
RANDOM_STATE = 42
TAG = "model_00_baseline"
LABELS = ["STOP", "FORWARD", "BACKWARD"]


def load_dataset(csv_path, img_size=IMG_SIZE):
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


def run_experiment(results_dir="results"):
    out = os.path.join(results_dir, TAG)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config_used.txt"), "w") as f:
        f.write(f"# {TAG} — BASELINE no-deep (model_00_baseline.py)\n\n")
        f.write(f"modelo = RandomForest (clasif. dirección + regresión velocidad, por motor)\n")
        f.write(f"img_size = {IMG_SIZE} (grises, aplanadas)\n")
        f.write(f"n_estimators = {N_ESTIMATORS}\nmax_depth = {MAX_DEPTH}\n")
        f.write(f"class_weight = balanced (dirección)\nrandom_state = {RANDOM_STATE}\n")

    X_tr, dirA_tr, dirB_tr, spdA_tr, spdB_tr = load_dataset(TRAIN_CSV)
    X_va, dirA_va, dirB_va, spdA_va, spdB_va = load_dataset(VAL_CSV)
    print(f"\nSplit por sesión -> Train: {len(X_tr)} | Val: {len(X_va)}")

    report_lines = []
    cms = {}
    # --- DIRECCIÓN: un Random Forest por motor ---
    for name, y_tr, y_va in [("A (IZQ)", dirA_tr, dirA_va), ("B (DER)", dirB_tr, dirB_va)]:
        print(f"\nEntrenando RF dirección Motor {name}...")
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                     random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight='balanced')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)
        rep = classification_report(y_va, y_pred, labels=LABELS, zero_division=0)
        report_lines.append(f"===== DIRECCIÓN MOTOR {name} =====\n{rep}")
        cms[name] = confusion_matrix(y_va, y_pred, labels=LABELS)
        print(rep)

    # --- VELOCIDAD: un regresor por motor ---
    for name, y_tr, y_va in [("A (IZQ)", spdA_tr, spdA_va), ("B (DER)", spdB_tr, spdB_va)]:
        print(f"\nEntrenando RF velocidad Motor {name}...")
        reg = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                    random_state=RANDOM_STATE, n_jobs=-1)
        reg.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_va, reg.predict(X_va))
        report_lines.append(f"MAE velocidad Motor {name}: {mae:.2f} (escala 0-100)")
        print(f"MAE velocidad Motor {name}: {mae:.2f}")

    # guardar report.txt
    report = "\n".join(report_lines)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(report + "\n")

    # guardar matrices de confusión por motor (mismo formato que los modelos deep)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ["A (IZQ)", "B (DER)"]):
        sns.heatmap(cms[name], annot=True, fmt='d', cmap='Blues',
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(f"Confusión — Motor {name}"); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    fig.suptitle("model_00_baseline — Random Forest (no-deep)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "eval_confusion.png"), dpi=150); plt.close(fig)
    print(f"\n✅ {TAG} listo. Resultados en {out}/")


if __name__ == "__main__":
    run_experiment()

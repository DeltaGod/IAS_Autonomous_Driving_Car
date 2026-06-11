"""
model_01_perframe.py — Modelo 1: PER-FRAME (MobileNetV3-small congelado).
=========================================================================
El MEJOR modelo hasta ahora. F1 macro dirección (val por sesión) = 0.629.
Una sola imagen -> 4 cabezas (dirA, dirB, spdA, spdB). Baseline a superar.

Hiperparámetros CONGELADOS abajo (esto ES el modelo). Motor en nn_perframe.py.
Correr:  .venv/bin/python model_01_perframe.py   ->  resultados en results/model_01_perframe/
"""
from config import Config
from nn_perframe import run_experiment

CFG = Config(
    # arquitectura
    neck_hidden=[256], dropout=0.4, backbone_frozen=True,
    # optimización
    batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=15,
    scheduler_step=7, scheduler_gamma=0.1,
    # balanceo / loss
    use_class_weights=True, lambda_speed=1.0,
)

if __name__ == "__main__":
    run_experiment(CFG, "model_01_perframe")

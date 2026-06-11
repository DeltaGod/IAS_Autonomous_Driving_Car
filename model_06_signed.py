"""
model_06_signed.py — Modelo 6: PER-FRAME con VELOCIDAD CON SIGNO.
==================================================================
Reframe del target: en vez de (dirección 3-clases + velocidad) por motor, el modelo
predice UNA variable continua por motor — el PWM con signo ∈ [-100,100]:
    > 0 FORWARD   < 0 BACKWARD   = 0 STOP

Misma arquitectura/optimización que el per-frame (model_01) para comparar parejo; lo
único distinto es el target (2 salidas continuas) y la loss (L1 ponderada por dirección
real de cada motor, para que BACKWARD no se muera al no haber class-weights).

Lee las bases SIGNED (generadas por build_signed_csv.py). Hiperparámetros CONGELADOS abajo.
Correr:  .venv/bin/python model_06_signed.py   ->  resultados en results/model_06_signed/
"""
from config import Config
from nn_signed import run_experiment

CFG = Config(
    # bases SIGNED (NO las viejas)
    train_csv="dataset_train_signed.csv",
    val_csv="dataset_val_signed.csv",
    # arquitectura (igual que model_01_perframe, para comparar parejo)
    neck_hidden=[256], dropout=0.4, backbone_frozen=True,
    # optimización
    batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=15,
    scheduler_step=7, scheduler_gamma=0.1,
    # balanceo: peso por muestra en la L1 según dirección (que BACKWARD sobreviva)
    use_class_weights=True,
)

if __name__ == "__main__":
    run_experiment(CFG, "model_06_signed")

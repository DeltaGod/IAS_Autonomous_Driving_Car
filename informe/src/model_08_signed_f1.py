"""
model_08_signed_f1.py -- Modelo 8: SIGNED (= model_06) pero F1 + pesos suavizados.
==================================================================================
Igual que model_06_signed (per-frame, PWM con signo) salvo DOS cambios, para arreglar
el colapso de STOP que tuvo model_06:
  1) SELECCION por F1 macro de direccion (en vez de MAE-PWM): la MAE elegia la peor
     epoca para direccion/seguridad (eligio la epoca 0; la 4 tenia mejor F1 y full-stop).
  2) PESOS SUAVIZADOS: class_weight_power=0.5 (raiz) -> backward ~6x baja a ~2.5x. El
     6x empujaba al modelo a comprometerse con un signo y aplastaba STOP (recall 0.08).

Hiperparametros CONGELADOS abajo (= model_06 + estos 2 cambios). Motor en nn_signed.py.
Correr:  .venv/bin/python model_08_signed_f1.py   ->  results/model_08_signed_f1/
"""
from config import Config
from nn_signed import run_experiment

CFG = Config(
    # bases SIGNED (identicas a model_06)
    train_csv="dataset_train_signed.csv",
    val_csv="dataset_val_signed.csv",
    # arquitectura/optimizacion IDENTICAS a model_06
    neck_hidden=[256], dropout=0.4, backbone_frozen=True,
    batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=15,
    scheduler_step=7, scheduler_gamma=0.1,
    # balanceo: pesos por direccion SUAVIZADOS (backward deja de dominar)
    use_class_weights=True,
    class_weight_power=0.5,
)

if __name__ == "__main__":
    # mejor epoca elegida por F1 macro (no por MAE)
    run_experiment(CFG, "model_08_signed_f1", select_by="f1")

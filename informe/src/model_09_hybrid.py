"""
model_09_hybrid.py -- Modelo 9: PER-FRAME HIBRIDO (signed + cabeza "is-stop").
=============================================================================
Intenta rescatar el STOP que el enfoque signed (model_06/08) no podia detectar.
Por motor: salida signed (tanh, PWM con signo) PARA cuando se mueve + cabeza binaria
"is-stop" (BCE) que decide explicitamente si esta parado.
  DECODE: STOP si is-stop>0.5 ; si no, direccion = signo(signed), speed = |signed|.

Igual que model_08 (bases signed, arquitectura per-frame, pesos suavizados) + la cabeza
is-stop con su BCE (lambda_stop). Seleccion por F1 macro. Motor en nn_hybrid.py.
Correr:  .venv/bin/python model_09_hybrid.py   ->  results/model_09_hybrid/
"""
from config import Config
from nn_hybrid import run_experiment

CFG = Config(
    # bases SIGNED (mismas que model_06/08)
    train_csv="dataset_train_signed.csv",
    val_csv="dataset_val_signed.csv",
    # arquitectura/optimizacion (= model_08)
    neck_hidden=[256], dropout=0.4, backbone_frozen=True,
    batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=15,
    scheduler_step=7, scheduler_gamma=0.1,
    # balanceo
    use_class_weights=True,
    class_weight_power=0.5,   # pesos L1 por direccion suavizados (= model_08)
    lambda_stop=1.0,          # peso de la BCE de la cabeza is-stop
)

if __name__ == "__main__":
    run_experiment(CFG, "model_09_hybrid")

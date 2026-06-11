"""
model_02_gru_v1.py — Modelo 2: SECUENCIAL GRU v1 (sin regularizar).
====================================================================
Primer intento temporal. F1 macro dirección (val) = 0.514. Overfit desde la época 1
(STOP recall colapsó a ~0.26: el modelo se refugió en FORWARD). Quedó por DEBAJO del
per-frame (0.629). Sirve como punto de partida del experimento temporal.

Hiperparámetros CONGELADOS abajo (esto ES el modelo). Motor en nn_sequence.py.
Correr:  .venv/bin/python model_02_gru_v1.py   ->  resultados en results/model_02_gru_v1/
"""
from config import Config
from nn_sequence import run_experiment

CFG = Config(
    # red recurrente (sin regularizar: hidden grande, dropout/wd bajos)
    rnn_type="GRU", rnn_hidden=128, rnn_layers=1,
    neck_hidden=[256], dropout=0.4, weight_decay=1e-4,
    # ventana temporal (~1s de contexto a 22fps)
    seq_len=8, seq_stride=3, seq_window_step=1,
    # optimización (batch chico por la RAM unificada de la Jetson)
    batch_size=16, lr=1e-3, max_epochs=15,
    scheduler_step=7, scheduler_gamma=0.1,
    use_class_weights=True, lambda_speed=1.0,
)

if __name__ == "__main__":
    run_experiment(CFG, "model_02_gru_v1")

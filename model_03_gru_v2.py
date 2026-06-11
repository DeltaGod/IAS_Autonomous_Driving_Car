"""
model_03_gru_v2.py — Modelo 3: SECUENCIAL GRU v2 (regularizado).
=================================================================
GRU con MÁS regularización para frenar el overfit de v1: hidden más chico (64),
dropout 0.5, weight_decay 1e-3 y 25 épocas. F1 macro dirección (val) = 0.554 (mejor
de los temporales). Recuperó el STOP recall vs v1, pero igual NO supera al per-frame.

Hiperparámetros CONGELADOS abajo (esto ES el modelo). Motor en nn_sequence.py.
Correr:  .venv/bin/python model_03_gru_v2.py   ->  resultados en results/model_03_gru_v2/
"""
from config import Config
from nn_sequence import run_experiment

CFG = Config(
    # red recurrente REGULARIZADA (hidden chico, dropout/wd altos)
    rnn_type="GRU", rnn_hidden=64, rnn_layers=1,
    neck_hidden=[256], dropout=0.5, weight_decay=1e-3,
    # ventana temporal (~1s de contexto a 22fps)
    seq_len=8, seq_stride=3, seq_window_step=1,
    # optimización
    batch_size=16, lr=1e-3, max_epochs=25,
    scheduler_step=12, scheduler_gamma=0.1,
    use_class_weights=True, lambda_speed=1.0,
)

if __name__ == "__main__":
    run_experiment(CFG, "model_03_gru_v2")

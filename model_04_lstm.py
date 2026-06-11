"""
model_04_lstm.py — Modelo 4: SECUENCIAL LSTM (= config de GRU v2, celda LSTM).
==============================================================================
Misma config regularizada que GRU v2 pero con celda LSTM (rnn_type="LSTM").
F1 macro dirección (val) = 0.544 ≈ GRU v2 (0.554), dentro del ruido. Confirma que
GRU vs LSTM es decisión de 2º orden: el LSTM tiene +33% de params (más overfit) y a
seq_len=8 su memoria larga no se aprovecha. Tampoco supera al per-frame.

Hiperparámetros CONGELADOS abajo (esto ES el modelo). Motor en nn_sequence.py.
Correr:  .venv/bin/python model_04_lstm.py   ->  resultados en results/model_04_lstm/
"""
from config import Config
from nn_sequence import run_experiment

CFG = Config(
    # IDÉNTICA a GRU v2 salvo el tipo de celda recurrente
    rnn_type="LSTM", rnn_hidden=64, rnn_layers=1,
    neck_hidden=[256], dropout=0.5, weight_decay=1e-3,
    seq_len=8, seq_stride=3, seq_window_step=1,
    batch_size=16, lr=1e-3, max_epochs=25,
    scheduler_step=12, scheduler_gamma=0.1,
    use_class_weights=True, lambda_speed=1.0,
)

if __name__ == "__main__":
    run_experiment(CFG, "model_04_lstm")

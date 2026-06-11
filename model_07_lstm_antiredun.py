"""
model_07_lstm_antiredun.py — Modelo 7: LSTM + ANTI-REDUNDANCIA.
================================================================
IDÉNTICO a model_05_gru_v3_antiredun (GRU v2 regularizado + seq_window_step=4) pero
con celda LSTM en vez de GRU. Completa el grid GRU/LSTM × (con/sin anti-redundancia):
  model_03 GRU v2  |  model_04 LSTM (=v2)  |  model_05 GRU+antiredun  |  model_07 LSTM+antiredun
Target = dirección 3-clases + velocidad por motor (nn_sequence), igual que model_05.

Hiperparámetros CONGELADOS abajo (= model_05, solo cambia rnn_type). Motor en nn_sequence.py.
Correr:  .venv/bin/python model_07_lstm_antiredun.py  ->  results/model_07_lstm_antiredun/
"""
from config import Config
from nn_sequence import run_experiment

CFG = Config(
    # IDÉNTICA a model_05 salvo el tipo de celda recurrente (GRU -> LSTM)
    rnn_type="LSTM", rnn_hidden=64, rnn_layers=1,
    neck_hidden=[256], dropout=0.5, weight_decay=1e-3,
    seq_len=8, seq_stride=3, seq_window_step=4,   # anti-redundancia (= model_05)
    batch_size=16, lr=1e-3, max_epochs=25,
    scheduler_step=12, scheduler_gamma=0.1,
    use_class_weights=True, lambda_speed=1.0,
)

if __name__ == "__main__":
    run_experiment(CFG, "model_07_lstm_antiredun")

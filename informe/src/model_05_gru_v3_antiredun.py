"""
model_05_gru_v3_antiredun.py -- Modelo 5: GRU v2 + ANTI-REDUNDANCIA.
====================================================================
Misma config que GRU v2 pero con seq_window_step=4: en TRAIN se queda con 1 de cada 4
ventanas para cortar el solape entre secuencias casi identicas (sliding window a 22fps).
Ultima palanca "entrenable" sin tocar el dataset. F1 macro direccion (val) = 0.550:
NO mejoro sobre GRU v2 (0.554), pero dio el MEJOR MAE de velocidad (~17). Confirma el
veredicto: el techo es el DATO, no la arquitectura.

Hiperparametros CONGELADOS abajo (esto ES el modelo). Motor en nn_sequence.py.
Correr:  .venv/bin/python model_05_gru_v3_antiredun.py  ->  results/model_05_gru_v3_antiredun/
"""
from config import Config
from nn_sequence import run_experiment

CFG = Config(
    # IDENTICA a GRU v2 + anti-redundancia de ventanas en train
    rnn_type="GRU", rnn_hidden=64, rnn_layers=1,
    neck_hidden=[256], dropout=0.5, weight_decay=1e-3,
    seq_len=8, seq_stride=3, seq_window_step=4,   # <- la diferencia con GRU v2
    batch_size=16, lr=1e-3, max_epochs=25,
    scheduler_step=12, scheduler_gamma=0.1,
    use_class_weights=True, lambda_speed=1.0,
)

if __name__ == "__main__":
    run_experiment(CFG, "model_05_gru_v3_antiredun")

# Resultados de los modelos — convención `model_NN_*`

Cada modelo entrenado tiene **un script entrypoint** `model_NN_<nombre>.py` (en la raíz del
repo) con sus **hiperparámetros congelados adentro**, y **una carpeta** `results/model_NN_<nombre>/`
con sus resultados. El nombre `model_NN_*` aparece igual en el código y en la carpeta, así
se ve de un vistazo qué código produjo qué resultado.

## Cómo está organizado el código

- **Entrypoints (un archivo = un modelo):** `model_01_perframe.py`, `model_02_gru_v1.py`,
  `model_03_gru_v2.py`, `model_04_lstm.py`, `model_05_gru_v3_antiredun.py`.
  Cada uno arma una `Config` congelada (los hiperparámetros que DEFINEN ese modelo, visibles
  al abrir el archivo) y llama a `run_experiment(CFG, "model_NN_...")`.
- **Motores compartidos (no se corren directo):**
  - `nn_perframe.py` — Dataset + `MotorControlNet` (per-frame) + eval/plots + `run_experiment`.
  - `nn_sequence.py` — Dataset de secuencias + `MotorControlGRU` (CNN+GRU/LSTM) + `run_experiment`.
    Reutiliza (hereda) la loss/métricas/eval de `nn_perframe`.
- `config.py` — define la dataclass `Config` (el esquema de parámetros). Los modelos la
  instancian con valores congelados; ya no hay un `CFG` global compartido entre experimentos.

Para re-entrenar cualquiera:  `.venv/bin/python model_03_gru_v2.py`
(detecta GPU/CPU solo; deja todo en `results/model_03_gru_v2/`).

## Qué hay en cada carpeta `results/model_NN_*/`

| archivo | qué es | en git |
|---|---|---|
| `eval_confusion.png` | matrices de confusión 3×3 por motor (mejor checkpoint) | ✅ |
| `training_progress.png` | curvas: loss, F1 por motor, MAE velocidad, LR | ✅ |
| `metrics.csv` | métricas por época (CSVLogger de Lightning) | ✅ |
| `report.txt` | classification_report por motor + MAE | ✅ |
| `config_used.txt` | volcado de los hiperparámetros congelados usados | ✅ |
| `model.ckpt` | pesos entrenados (mejor checkpoint) | ❌ (gitignored) |

Los pesos (`model.ckpt`) y los `logs/` de Lightning están gitignoreados (binarios pesados);
se versionan solo las métricas/curvas/reportes.

## Scoreboard (val por sesión)

Métrica principal = **F1 macro de dirección por motor** (3 clases). Velocidad = MAE (0-100).

| # | modelo | código | F1 macro (A / B) | STOP recall (A / B) | BACKWARD recall (A / B) | MAE (A / B) |
|---|---|---|---|---|---|---|
| 00 | baseline RF (no-deep) | `model_00_baseline.py` | 0.40 / 0.23 | 0.51 / 0.00 | 0.10 / 0.03 | 19.8 / 23.2 |
| 01 | **per-frame** ⭐ | `model_01_perframe.py` | **0.62 / 0.63** | 0.44 / 0.46 | 0.73 / 0.73 | 19.5 / 21.3 |
| 02 | GRU v1 (sin regul.) | `model_02_gru_v1.py` | 0.45 / 0.57 | 0.26 / 0.29 | 0.38 / 0.39 | 22.3 / 22.9 |
| 03 | GRU v2 (regul.) | `model_03_gru_v2.py` | 0.50 / 0.62 | 0.42 / 0.55 | 0.38 / 0.39 | 17.8 / 18.8 |
| 04 | LSTM (= v2) | `model_04_lstm.py` | 0.51 / 0.58 | 0.46 / 0.64 | 0.32 / 0.32 | 18.4 / 18.3 |
| 05 | GRU v3 anti-redun | `model_05_gru_v3_antiredun.py` | 0.50 / 0.60 | 0.42 / 0.70 | 0.39 / 0.38 | 17.0 / 18.1 |

**Modelo 00 = piso de comparación.** Es un Random Forest sobre píxeles crudos (grises 32×32),
SIN deep learning. Colapsa: en motor B no reconoce ni un STOP (F1 0.00) y casi ningún BACKWARD;
predice casi todo FORWARD. El salto de 0.40/0.23 (RF) a **0.62/0.63** (per-frame) es la prueba
de que la CNN aporta valor real, no es complejidad gratis. El baseline NO tiene `metrics.csv`
(no entrena por épocas) ni `model.ckpt` (es sklearn, no Lightning).

⭐ **El per-frame (modelo 01) es el mejor de los deep y el baseline a superar.** Ningún temporal
lo supera en F1 global. Observaciones:

- **STOP↔FORWARD es el error dominante** en todos (STOP recall ~0.4-0.46 en el per-frame). Es
  el borde velocidad=0, físicamente ambiguo desde un frame.
- **BACKWARD**: lo maneja bien el per-frame (recall 0.73); en los temporales cae (0.32-0.39).
- Los temporales **regularizados** (v2/LSTM/v3) recuperan STOP en motor B (0.55-0.70) y dan
  **mejor MAE** (~17-18) que el per-frame (~20), pero pierden F1 global.

> ⚠️ **Aviso de comparabilidad:** el val per-frame tiene 686 frames; el val secuencial tiene
> 519 ventanas (la ventana etiqueta el ÚLTIMO frame y descarta los primeros de cada record),
> con distribución de clases distinta (p.ej. BACKWARD 26% per-frame vs 13% secuencial). Las
> columnas de arriba son comparables DENTRO de cada familia; entre familias, mirar tendencias.

**Veredicto:** el techo es el DATO, no la arquitectura. Detalle en
`memory/temporal-model-results.md`.

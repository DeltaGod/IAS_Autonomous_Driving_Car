# Resultados de los modelos — convención `model_NN_*`

Cada modelo entrenado tiene **un script entrypoint** `model_NN_<nombre>.py` (en la raíz del
repo) con sus **hiperparámetros congelados adentro**, y **una carpeta** `results/model_NN_<nombre>/`
con sus resultados. El nombre `model_NN_*` aparece igual en el código y en la carpeta, así
se ve de un vistazo qué código produjo qué resultado.

## Cómo está organizado el código

- **Entrypoints (un archivo = un modelo):** `model_00_baseline.py` … `model_09_hybrid.py`.
  Cada uno arma una `Config` congelada (los hiperparámetros que DEFINEN ese modelo, visibles
  al abrir el archivo) y llama a `run_experiment(CFG, "model_NN_...")`.
- **Motores compartidos (no se corren directo):**
  - `nn_perframe.py` — Dataset + `MotorControlNet` (per-frame, dir 3-clases + velocidad) + eval/plots.
  - `nn_sequence.py` — Dataset de secuencias + `MotorControlGRU` (CNN+GRU/LSTM). Hereda de `nn_perframe`.
  - `nn_signed.py` — `SignedControlNet` (per-frame, target = PWM CON SIGNO ∈[-100,100], 2 salidas
    continuas). Loss = L1 ponderada por dirección; métrica nativa = MAE-PWM-con-signo + full-stop recall.
  - `nn_hybrid.py` — `HybridControlNet` (per-frame, signed + cabeza binaria "is-stop" por motor).
    Loss = L1(signed) + λ·BCE(is-stop, pos_weight por motor). Decode: STOP si is-stop>0.5, si no signo(signed).
- **Generadores de datos:** `build_global_csv.py` (dir+speed → `dataset_train.csv`/`dataset_val.csv`),
  `build_signed_csv.py` (envuelve al anterior y produce el target con signo → `dataset_*_signed.csv`).
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
| 06 | signed (sel. MAE) † | `model_06_signed.py` | 0.43 / 0.54 | **0.08 / 0.09** | **0.76 / 0.74** | 16.4 / 21.7 |
| 07 | LSTM anti-redun | `model_07_lstm_antiredun.py` | 0.52 / 0.56 | 0.50 / 0.55 | 0.30 / 0.30 | 17.3 / 18.7 |
| 08 | signed (sel. F1, pesos suaves) † | `model_08_signed_f1.py` | 0.42 / 0.50 | **0.07 / 0.08** | 0.63 / 0.58 | 17.7 / 21.5 |
| 09 | híbrido signed + is-stop † | `model_09_hybrid.py` | 0.45 / 0.33 | 0.13 / **0.96** | 0.53 / 0.20 | 18.4 / 22.6 |

† **Modelos signed (06, 08):** el target es PWM con signo (1 salida continua/motor), no clasificación.
La dirección de la tabla se **decodifica** del signo (umbral 1 PWM) para comparar; la columna MAE es
**MAE-PWM-con-signo** (no velocidad). Tienen además **full-stop recall** (model_06: 0.025). Detalle abajo.

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

### Experimentos temporales y signed (modelos 06-08)

- **07 (LSTM anti-redun)** completa el grid GRU/LSTM × redundancia: F1 0.52/0.56, igual que el
  GRU equivalente (05). Reconfirma: GRU≈LSTM, ningún temporal supera al per-frame.
- **06 (signed, target = PWM con signo)** probó unificar dirección+velocidad en una salida continua.
  Resultado revelador: **BACKWARD sobrevivió** (recall 0.76/0.74, gracias a la L1 ponderada) pero
  **STOP se desplomó** (recall 0.08, full-stop 0.025). Detectar STOP = que la salida caiga en ±1 de 0,
  y la MAE no premia pegarle al cero → el modelo se "compromete con un signo". Además la **selección
  por MAE eligió la peor época** para dirección (la época 0; la 4 tenía mejor F1 y full-stop) — ejemplo
  de manual de por qué la métrica de selección importa.
- **08 (signed + F1 + pesos suavizados)** atacó eso: selección por F1 macro y `class_weight_power=0.5`
  (backward 6×→2.5×). **No funcionó: STOP siguió muerto** (recall 0.07/0.08, full-stop 0.025) y suavizar
  el backward solo le bajó su recall (0.76→0.63) sin rescatar STOP. **Conclusión: el colapso de STOP en
  el modelo signed es ESTRUCTURAL** (la regresión-a-cero no se detecta por umbral; ninguna ponderación
  ni criterio de selección lo arregla). El fix real sería arquitectónico: salida signed + una cabeza
  auxiliar "is-stop" (híbrido).
- **09 (híbrido signed + is-stop)** valida ese fix: la cabeza binaria explícita **SÍ detecta el cero** y
  sube el full-stop recall de **0.025 → 0.21** (8×) — el colapso de STOP del signed **es solucionable**.
  PERO la primera config **se pasa de rosca**: el `pos_weight` del motor B (6.5×, porque casi nunca para)
  hace que su cabeza is-stop **sobre-prediga STOP** (recall 0.96) y mate FORWARD (0.12) → el F1 macro cae a
  0.39 (el más bajo). El mecanismo funciona; falta **calibrarlo** (bajar `lambda_stop`/`pos_weight`).
  Pasó de problema ESTRUCTURAL a problema de TUNING.

Para esta tarea, la **clasificación explícita de dirección del per-frame (01) sigue siendo superior** a
todo el árbol signed/temporal.

Las palancas `class_weight_power` y `lambda_stop` (config.py), `build_signed_csv.py`, `nn_signed.py` y
`nn_hybrid.py` son los aportes de código de este experimento (los viejos quedaron intactos).

**Veredicto:** el techo es el DATO, no la arquitectura. Detalle en
`memory/temporal-model-results.md`.

## Exploración de hiperparámetros — `sweep.py`

`sweep.py` es un explorador GENÉRICO (sirve para cualquier motor) que barre una grilla,
corre cada config en un **subproceso aislado** (evita acumular memoria CUDA), y deja todo
en `results/sweeps/<nombre>/` (NO pisa los `results/model_NN/`). Produce:
- `sweep_summary.csv` (1 fila/config, métricas del mejor checkpoint),
- `pareto.png` (frontera de **Pareto F1-macro vs full-stop-recall** — porque el F1 solo no
  protege el frenado: rankea al 08 que no frena sobre el 09 que sí),
- `param_effects.png` (efecto marginal de cada parámetro),
- `reseed_summary.csv` (**anti-ruido**: re-corre los finalistas con varias seeds → media±desvío,
  para ver si la ventaja sobrevive al ruido del val ±0.05).

Sweeps definidos (editar la grilla en la función correspondiente):
- `model09_sweep()` → `m09_tier2`: 36 configs del híbrido (`lambda_stop`×`stop_pos_weight_cap`×`neck_hidden`×`dropout`).
- `model01_sweep()` → `m01_perframe`: 24 configs del per-frame (`neck_hidden`×`dropout`×`lr`×`weight_decay`).

Correr: `python3 sweep.py --which m09` (o `m01`). Smoke rápido: `--smoke`.

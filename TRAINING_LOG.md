# 📓 Bitácora de Entrenamiento (Training Log)

Este documento registra la evolución del modelo de Deep Learning para el proyecto IAS Autonomous Driving Car. Cada entrada detalla la configuración, los resultados y las lecciones aprendidas. El logging es **local** (CSVLogger de Lightning + matplotlib); no se usan servicios externos.

> **Nota de esquema:** las versiones previas (V1.0–V1.2) entrenaban un **clasificador global de 5 clases** (FORWARD/BACKWARD/LEFT/RIGHT/STOP) con logging en Weights & Biases y métrica de *accuracy*. Ese enfoque fue **retirado**: el modelo ahora predice **control por motor** (4 salidas). Las entradas viejas se eliminaron por no corresponder al esquema vigente.


## [V2.0] Reformulación Multi-Salida por Motor
**Fecha:** 8 de Junio, 2026
**Estado:** Vigente (arquitectura actual)

### Estrategia y Cambios
1. **Tarea redefinida:** IMAGEN → **4 salidas**, por motor IZQ (A) y DER (B):
   - `behaviorA`, `behaviorB` ∈ {STOP, FORWARD, BACKWARD} (clasificación, 3 clases c/u).
   - `speedA`, `speedB` ∈ [0,100] (regresión, normalizada a [0,1] con sigmoide + L1).
   El modelo NO predice GPIO ni el behavior global de 5 clases (se reconstruyen en el harness `predictions_to_control`).
2. **Pérdida combinada:** `CE(dirA) + CE(dirB) + λ·(L1(spdA) + L1(spdB))`, con `λ=1.0`.
3. **Balanceo:** `class_weights` inversos a la frecuencia (pooled A+B) **reincorporados** en la CE de dirección. El desbalance se atenúa además porque, por motor, la clase rara pasa a ser BACKWARD (~5.5%) en lugar del casi inexistente RIGHT global (0.4%).
4. **Augmentation flip:** en **espacio abstracto** (swap motor A↔B de velocidad y dirección). NO se swapean bits GPIO (la convención asimétrica invertiría forward↔backward).
5. **Transfer Learning:** `MobileNetV3-Small` con *feature extractor* completamente congelado; se entrena cuello compartido + 4 cabezas. Sin recortes espaciales (se descartó el top-crop, que en V1.2 cegó al modelo ante el punto de fuga).
6. **Métrica de optimización:** **F1 macro por motor** (3 clases) calculado acumulando toda la época (no promedio por batch) + MAE de velocidad. Checkpoint por `val_dir_f1_macro`.
7. **Split por sesión:** train = 1ª sesión (24.490 frames), val = otras 2 sesiones (686 frames).

### Resultados
* **Estado:** Entrenado. Es el **mejor modelo** (`model_01_perframe`).
* **Val F1 macro (Motor A / B):** **0.62 / 0.63** (≈0.629).
* **Val MAE velocidad (A / B):** 19.5 / 21.3.
* **STOP recall (A / B):** 0.44 / 0.46 · **BACKWARD recall:** 0.73 / 0.73.

### Diagnóstico Técnico
* **Imbalance disuelto (hipótesis):** descomponer la decisión por motor debería eliminar el problema del RIGHT inaprendible (0.4%), repartiéndolo entre direcciones por motor más balanceadas.
* **Riesgo de regresión sintética:** el 91% de las velocidades son interpoladas (rampas lineales) y casi bimodales (0 / ~60). El MAE de velocidad debe leerse con cautela; evaluar si conviene pasar a niveles discretos.
* **Riesgo de generalización:** val proviene de sesiones distintas a train, con distribución muy diferente (más BACKWARD/STOP). Es el test honesto de generalización entre sesiones.


## [V3.0] Árbol de experimentos + convención `model_NN`
**Fecha:** 13 de Junio, 2026
**Estado:** Vigente

El código se reorganizó a **un entrypoint por modelo** (`model_00_baseline` … `model_09_hybrid`) sobre motores compartidos (`nn_perframe`/`nn_sequence`/`nn_signed`/`nn_hybrid`). **Scoreboard completo y análisis en `results/README.md`.** Resumen:

- **Baseline (00):** Random Forest sobre píxeles 32×32 → F1 0.40/0.23. Piso: justifica el deep learning.
- **Temporales (02-05, 07):** CNN+GRU/LSTM sobre secuencias de 8 frames. Ninguno supera al per-frame (~0.51-0.55); overfit, no falta de capacidad. El techo es el DATO.
- **Signed (06, 08):** target = PWM con signo (1 salida continua/motor). Unifica el comando físico pero **el STOP colapsa estructuralmente** (la regresión no detecta el cero por umbral).
- **Híbrido (09):** signed + cabeza binaria "is-stop" (BCE). **Recupera el STOP** (full-stop 0.025→0.21) pero se descalibra (sobre-para motor B) → F1 0.39. De estructural a tuneable.
- **`sweep.py`:** explorador genérico (Pareto F1-vs-full-stop + reseed anti-ruido). Sweeps `m09_tier2` (híbrido) y `m01_perframe` (per-frame) corriendo en la Jetson.

**Conclusión:** el **per-frame con clasificación explícita (V2.0 / model_01) sigue siendo el mejor**. Ni temporal ni signed/híbrido lo superan; el límite es el dato (91% etiquetas interpoladas, 1 sola sesión, val chico).

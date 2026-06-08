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
* **Estado:** Pendiente — primera corrida del esquema V2.0 no ejecutada aún.
* **Val F1 macro (Motor A):** _por correr_
* **Val F1 macro (Motor B):** _por correr_
* **Val MAE velocidad (A / B):** _por correr_
* **Val Loss / Train Loss:** _por correr_

### Diagnóstico Técnico
* **Imbalance disuelto (hipótesis):** descomponer la decisión por motor debería eliminar el problema del RIGHT inaprendible (0.4%), repartiéndolo entre direcciones por motor más balanceadas.
* **Riesgo de regresión sintética:** el 91% de las velocidades son interpoladas (rampas lineales) y casi bimodales (0 / ~60). El MAE de velocidad debe leerse con cautela; evaluar si conviene pasar a niveles discretos.
* **Riesgo de generalización:** val proviene de sesiones distintas a train, con distribución muy diferente (más BACKWARD/STOP). Es el test honesto de generalización entre sesiones.

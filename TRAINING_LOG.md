# 📓 Bitácora de Entrenamiento (Training Log)

Este documento registra la evolución del modelo de Deep Learning para el proyecto IAS Autonomous Driving Car. Cada entrada detalla la configuración, los resultados obtenidos en Weights & Biases y las lecciones aprendidas.

---

## [V1.1] Fine-Tuning Parcial y Balanceo
**Fecha:** 8 de Junio, 2026  
**Estado:** Actual (Mejor Modelo)

### Estrategia y Cambios
1. **Fine-Tuning:** Descongelamiento de los bloques convolucionales 10, 11 y 12 de `MobileNetV3-Small`.
2. **Balanceo de Clases:** Implementación de pesos dinámicos en `CrossEntropyLoss` basados en la frecuencia inversa de las muestras.
3. **Regularización:** Incremento de Dropout a `0.5` en el clasificador.
4. **Optimización:** Reducción del Learning Rate a `5e-4` con `StepLR`.

### Resultados (Época 9)
*   **Val Accuracy:** ~59.6% (📈 +6% respecto a V1.0)
*   **Val Loss:** 1.522
*   **Train Loss:** 0.168
*   **ID de Run (W&B):** `run-20260607_205919-ufi5waks`

### Diagnóstico Técnico
*   **Éxito en Balanceo:** La caída del *accuracy* de entrenamiento comparado con la V1.0 es un indicador de que el modelo ya no está sesgado hacia la clase mayoritaria (FORWARD).
*   **Generalización:** El descongelamiento de las capas finales permitió al modelo aprender la semántica específica de la pista de conducción.
*   **Punto Crítico:** Existe un sobreajuste por memorización de las clases minoritarias debido a la alta capacidad del modelo y la agresividad de los pesos.

---

## [V1.0] Baseline Inicial
**Fecha:** 7 de Junio, 2026  
**Estado:** Obsoleto

### Estrategia
*   Transfer Learning con `MobileNetV3-Small` pre-entrenado en ImageNet.
*   *Feature Extractor* completamente congelado.
*   Entrenamiento solo de la capa de salida (*Head*) por 10 épocas.

### Resultados
*   **Val Accuracy:** 53.85%
*   **Train Accuracy:** 65.90%
*   **Val Loss:** 1.364
*   **ID de Run (W&B):** `run-20260607_202852-haah3sui`

### Diagnóstico Técnico
*   **Sesgo:** El modelo predice principalmente la clase mayoritaria.
*   **Limitación de Dominio:** Los filtros de ImageNet no son suficientes para el control lateral y detección de pista sin un mínimo de fine-tuning.
*   **Overfitting:** El modelo dejó de aprender en la época 0 (mínimo de Val Loss alcanzado inmediatamente).

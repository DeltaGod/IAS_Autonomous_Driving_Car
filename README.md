# IAS_Autonomous_Driving_Car
IAS - ENIB : School Proyect to program an autonomus driving car

# Instalar dependencias desde el requirements
pip install -r requirements.txt


# Cosas a hacer:

Pretratamiento CSV:
1. Poner en 0 las velocidades en las entradas donde los GPIO estén todos en 0
2. Generar data artificial (usando interpolaciones) para poder asociarle una entrada del .csv a cada imagen
3. Asociar entradas del .csv con su imágen correspondiente


# Propuesta:
Creamos un script de python que genere un nuevo CSV global que reuna la informacion de las carpetas de la base de datos, entre los csv y las imagenes. El formato sera igual que el de los CSV con una columna con las direcciones asociadas a cada imagen.

Si alguna fila dice que TODOS sus GPIO estan en 0, y la velocidad esta en un valor distinto de 0 ya sea para motor A o motor B, es un error del codigo y debera ser corregido cambiando la velocidad de motor A y motor B a 0.
Esto es asi ya que el dirver internamente FRENA el motor si todos los GPIO de sentido estan en 0.

 Arrancamos asociando la imagen anterior de un comando mas cercana al momento del comando. Dado que existen muchas mas imagenes que comandos CSV van a quedar imagenes no asociadas a ningun comando. Para estas imagenes no asociadas vamos a interpolar todos los campos del comando (slavo el tiempo) entre comando real y comando real, dependiendo de cuantas imagenes sin asociar existen entre estos 2 comandos. Utilizaremos el tiempo de la imagen como parametro de entrada para interpolar los campos. Es decir tengo una imagen llamada 1 (tomada en el ms 1) y tenemos un comando hecho en ms 2, el proximo comando esta hecho en ms 5. Existen imagen 1, imagen 3 imagen 4. El comando hecho en ms 2 es asociado a la imagen anterior (es decir 1), imagen 4 es asociada al comando de ms 5. imagen 3 no tiene comando asociado asique interpolamos todos los parametros del comando utilizando el comando en ms 2 y en ms 5 como parametros de borde para la interpolacion. 

La database esta MUY poblada por curvas a la IZQUIERDA, Se propone flipear en entrenamiento imagenes at random


## 🚀 Entrenamiento y Modelos

El entrenamiento se gestiona mediante `train_mobilenet.py` con logging integrado en **Weights & Biases**. 

*   **Modelo Actual:** MobileNetV3-Small (V1.1)
*   **Métrica Principal:** ~60% Val Accuracy.

Para un análisis detallado de la evolución del modelo, hiperparámetros y diagnósticos de cada versión, consultá la bitácora completa:

👉 **[TRAINING_LOG.md](./TRAINING_LOG.md)**

## 📐 Registro de Decisiones Técnicas (Design Log)

Esta sección documenta las decisiones críticas tomadas durante el diseño del *pipeline* de datos y la arquitectura del modelo. Sirve como bitácora de depuración: si el comportamiento del vehículo en inferencia no es el esperado, estos son los puntos de control a revisar.

### 1. Procesamiento de la Base de Datos (Data Pipeline)

| Componente | Decisión Tomada | Justificación | Riesgo / Fallback |
| :--- | :--- | :--- | :--- |
| **Generación de Comandos** | **Interpolación Lineal** de valores PWM para imágenes sin comando directo. | Asegurar que cada *frame* tenga un comando asociado sin perder volumen de datos. | **Riesgo:** El modelo puede aprender a acelerar/frenar gradualmente (latencia artificial) en lugar de ser reactivo.<br>**Fallback:** Cambiar a *Forward-Fill* estricto. |
| **Gestión de Freno (GPIO)** | **Retención de estado:** Los pines GPIO mantienen su dirección mientras la velocidad interpolada sea `> 0`. | Evita clasificar la desaceleración inercial como un estado de `STOP` absoluto, lo que sesgaría al modelo a no moverse. | N/A |
| **Control de Latencia** | **Segmentación de Sesiones (*Chunking*):** Descarte de *frames* y corte de sesión si el $\Delta T > 1000ms$. | Previene el *Covariate Shift* temporal. Evita que la red intente aprender de saltos espaciales ("teletransportes"). | N/A |

### 2. Estrategia de Entrenamiento (Deep Learning)

| Componente | Decisión Tomada | Justificación | Riesgo / Fallback |
| :--- | :--- | :--- | :--- |
| **Balanceo de Clases** | **Mirroring Estocástico en *Runtime*:** Espejado de imágenes + inversión de comandos `speedA`/`speedB` y etiquetas. | Corrige el sesgo extremo del *dataset* original (giros a izquierda vs derecha) sin duplicar archivos en disco duro. | Verificar que no haya asimetrías físicas en el laboratorio que confundan a la red al ser espejadas. |
| **Transfer Learning** | **Entrenamiento en Dos Fases:** 1. Congelar *backbone* de MobileNetV3. 2. *Unfreeze* total con bajo *Learning Rate*. | Previene el *Catastrophic Forgetting*. Evita que gradientes iniciales altos destruyan los pesos pre-entrenados en ImageNet. | Si la red no converge, ajustar el *Learning Rate Scheduler* (ej. StepLR). |
| **Robustez Visual** | **Color Jittering Controlado:** Variación aleatoria de brillo y contraste durante la carga de *batches*. | Desvincula al modelo de las condiciones de iluminación exactas del momento de grabación. | Si se aplican valores muy altos, puede "quemar" la imagen y ocultar las líneas de la pista. Requiere *tuning* previo. |
| **Gestión Espacial de Imagen** | **Top-Cropping Paramétrico:** Recorte geométrico y estricto del tercio superior del tensor en la capa del `Dataset`. | Elimina el ruido de fondo (paredes, luces, objetos del laboratorio). Evita el *overfitting* de contexto, forzando a las capas convolucionales a reaccionar a la textura del asfalto. | N/A. Solo asegura que el recorte no elimine información temprana de la curva. |
| **Métrica de Optimización** | **Transición de Accuracy a F1-Score:** El optimizador y el guardado de *checkpoints* ahora priorizan el F1-Score de clases direccionales. | En bases de datos desbalanceadas con mucho movimiento recto (`FORWARD`), el *accuracy* engaña. El F1 castiga al modelo si ignora las curvas. | N/A. |
| **Balanceo de Función de Costo** | **Eliminación de Pesos Dinámicos:** Regreso a `CrossEntropyLoss` puro (peso 1.0 uniforme). | Los pesos multiplicaban los gradientes en errores de `RIGHT`. Como el *Mirroring* ya genera un balance 50/50 artificial de giros, los pesos generaban una sobre-penalización y corrompían el optimizador. | N/A. |
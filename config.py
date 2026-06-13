"""
config.py — PANEL DE CONTROL CENTRAL del proyecto.
==================================================
Todo lo configurable vive acá. El resto de los programas importan `CFG` y usan
estos valores (NO hardcodear nada de esto en otro lado).

  from config import CFG
  CFG.lr, CFG.batch_size, CFG.resample_factor, ...

Corré este archivo directo para VER el efecto de tus parámetros sobre la
distribución del dataset (antes vs después de los tratamientos):

  .venv/bin/python config.py        ->  genera 'augmentation_preview.png'

Así tocás un número acá, re-corrés, y ves en el gráfico cómo cambia el balance.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ===================== DATOS / RUTAS =====================
    train_csv: str = "dataset_train.csv"
    val_csv: str = "dataset_val.csv"
    image_size: int = 224

    # ============== SÍNTESIS / AUGMENTATION ==============
    # El flip horizontal espeja la imagen y hace SWAP motor A<->B:
    #   - convierte LEFT<->RIGHT (balancea giros)
    #   - FORWARD / BACKWARD / STOP NO cambian de etiqueta (son simétricos)
    # Probabilidad de aplicar flip, separada por tipo de frame:
    flip_prob_turns: float = 0.5      # frames LEFT/RIGHT  (ej: subir a 0.6 para balancear izq/der)
    flip_prob_straight: float = 0.5   # frames FORWARD/BACKWARD/STOP (solo variedad visual)

    # Factor de remuestreo por clase GLOBAL (cuántas muestras sintéticas generar):
    #   0.0 = eliminar la clase | <1 = submuestrear | 1 = igual | >1 = sobremuestrear
    # Ej: para "eliminar stop" -> "STOP": 0.0 ; para "aumentar stop" -> "STOP": 2.0
    resample_factor: dict = field(default_factory=lambda: {
        "FORWARD": 1.0, "LEFT": 1.0, "RIGHT": 1.0, "STOP": 1.0, "BACKWARD": 1.0,
    })

    # Color jitter (robustez a iluminación)
    jitter_brightness: float = 0.15
    jitter_contrast: float = 0.15
    jitter_saturation: float = 0.15
    jitter_hue: float = 0.05

    # ===================== PÉRDIDA / BALANCEO =====================
    use_class_weights: bool = True    # pesos inversos a la frecuencia en la CE de dirección
    # Suaviza esos pesos elevándolos a esta potencia: 1.0 = inverso a freq (extremo),
    # 0.5 = raíz cuadrada (modera la clase rara: backward ~6x -> ~2.5x), 0.0 = uniforme.
    class_weight_power: float = 1.0
    lambda_speed: float = 1.0         # peso de la regresión de velocidad en la loss total
    lambda_stop: float = 1.0          # (modelo híbrido) peso de la BCE de la cabeza "is-stop"
    stop_pos_weight_cap: float = 0.0  # (híbrido) tope al pos_weight de la cabeza is-stop (0=sin tope)

    # ===================== ARQUITECTURA DEL MODELO =====================
    backbone_frozen: bool = True      # congelar el feature extractor de MobileNetV3
    # Cuello compartido: una entrada por capa oculta. nº de capas = len(lista); neuronas = valores.
    #   [256]        -> 1 capa de 256
    #   [512, 256]   -> 2 capas (512 y 256)
    neck_hidden: list = field(default_factory=lambda: [256])
    dropout: float = 0.4

    # ===================== OPTIMIZACIÓN =====================
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    scheduler_step: int = 7
    scheduler_gamma: float = 0.1
    num_workers: int = 4
    seed: int = 42

    # ============== MODELO SECUENCIAL (CNN + GRU/LSTM) ==============
    # Lo usa train_gru.py. El backbone (MobileNetV3) va SIEMPRE congelado:
    # lo que se entrena es la red recurrente + cuello + 4 cabezas.
    # La ventana es MANY-TO-ONE: se predice la acción del ÚLTIMO frame de la
    # secuencia (causal y deployable: en el auto sería un buffer rodante).
    # seq_len y seq_stride son ADEMÁS el experimento de "frecuencia de muestreo"
    # que pide el enunciado: a ~22fps, stride=3 ≈ ~1s de contexto temporal.
    seq_len: int = 8          # nº de frames por secuencia (ventana temporal)
    seq_stride: int = 3       # paso entre frames de la ventana (1 = consecutivos)
    seq_window_step: int = 1  # anti-redundancia: subsamplea ventanas en TRAIN (1=todas; 4=1 de cada 4)
    rnn_type: str = "GRU"     # "GRU" o "LSTM"
    rnn_hidden: int = 128     # tamaño del estado oculto recurrente
    rnn_layers: int = 1       # nº de capas recurrentes apiladas


CFG = Config()


# =================================================================
# Resumen de parámetros activos (para anotar en los gráficos)
# =================================================================
def active_params_text(cfg=CFG):
    rf = ", ".join(f"{k}:{v:g}" for k, v in cfg.resample_factor.items())
    return (
        f"flip_turns={cfg.flip_prob_turns:g} | flip_straight={cfg.flip_prob_straight:g} | "
        f"class_weights={cfg.use_class_weights}\n"
        f"resample_factor → {rf}"
    )


# =================================================================
# Simulación de la distribución resultante (valor esperado)
# =================================================================
def simulate_distribution(df, cfg=CFG):
    """Calcula la distribución de clases ANTES y DESPUÉS de aplicar:
    (1) resample_factor por clase global, (2) flip estocástico (swap A<->B y L<->R).
    Devuelve un dict con porcentajes por vista: global / motorA / motorB."""
    import pandas as pd

    flipmap = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    grp = df.groupby(["behavior", "behaviorA", "behaviorB"]).size().reset_index(name="c")

    rows = []
    for _, r in grp.iterrows():
        gb, a, b, c = r["behavior"], r["behaviorA"], r["behaviorB"], r["c"]
        w = cfg.resample_factor.get(gb, 1.0) * c
        p = cfg.flip_prob_turns if gb in ("LEFT", "RIGHT") else cfg.flip_prob_straight
        rows.append((gb, a, b, w * (1.0 - p)))             # no flipeado
        rows.append((flipmap.get(gb, gb), b, a, w * p))    # flipeado: swap motores y L<->R
    sim = pd.DataFrame(rows, columns=["behavior", "behaviorA", "behaviorB", "w"])

    g_order = ["FORWARD", "LEFT", "RIGHT", "STOP", "BACKWARD"]
    m_order = ["STOP", "FORWARD", "BACKWARD"]

    def before(col, order):
        vc = df[col].value_counts(normalize=True).mul(100)
        return [float(vc.get(k, 0.0)) for k in order]

    def after(col, order):
        s = sim.groupby(col)["w"].sum()
        tot = s.sum()
        return [float(s.get(k, 0.0) / tot * 100) if tot > 0 else 0.0 for k in order]

    return {
        "global": (g_order, before("behavior", g_order), after("behavior", g_order)),
        "motorA": (m_order, before("behaviorA", m_order), after("behaviorA", m_order)),
        "motorB": (m_order, before("behaviorB", m_order), after("behaviorB", m_order)),
    }


def plot_preview(cfg=CFG, out_png="augmentation_preview.png"):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if not os.path.exists(cfg.train_csv):
        print(f"[ERROR] No existe {cfg.train_csv}. Corré build_global_csv.py primero.")
        return
    df = pd.read_csv(cfg.train_csv)
    dist = simulate_distribution(df, cfg)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Distribución del dataset de ENTRENAMIENTO: ANTES vs DESPUÉS de los tratamientos",
                 fontsize=15)
    titles = {"global": "Behavior GLOBAL (5 clases)",
              "motorA": "Dirección Motor A (IZQ)",
              "motorB": "Dirección Motor B (DER)"}

    for ax, key in zip(axes, ["global", "motorA", "motorB"]):
        order, before, after = dist[key]
        x = np.arange(len(order))
        ax.bar(x - 0.2, before, 0.4, label="Antes (crudo)", color="#4C72B0")
        ax.bar(x + 0.2, after, 0.4, label="Después (config)", color="#DD8452")
        for xi, (bv, av) in enumerate(zip(before, after)):
            ax.text(xi - 0.2, bv + 0.5, f"{bv:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + 0.2, av + 0.5, f"{av:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=30)
        ax.set_title(titles[key]); ax.set_ylabel("% del split"); ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    # Anotación: QUÉ valores de las variables produjeron estos resultados.
    fig.text(0.5, 0.005, "Parámetros activos:  " + active_params_text(cfg).replace("\n", "   |   "),
             ha="center", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_png, dpi=150)
    print(f"📊 Preview guardado en '{out_png}'")
    print("Parámetros activos:\n" + active_params_text(cfg))


if __name__ == "__main__":
    plot_preview()

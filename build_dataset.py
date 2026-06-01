"""
build_dataset.py
================
Construye el "dataset maestro" (dataset.csv) recorriendo las 8 sesiones de
DataSet/#Record_*/ y alineando cada IMAGEN con la accion de control que el
piloto comando mientras ese frame estaba en pantalla.

Salida: un unico CSV con columnas
    image_path , direction , speed_level , record , time_in_ms , n_control

  - direction  : Forward / Backward / Left / Right   (NO hay Stop, ver abajo)
  - speed_level: Low / Medium / High

NO HAY CLASE "STOP"
-------------------
Se verifico sobre las 13.544 filas de las 8 sesiones que NUNCA los dos motores
estan en velocidad 0 al mismo tiempo: el auto siempre esta en movimiento. Los
casos con un solo motor en 0 son giros pivote, y las filas con los 4 GPIO en 0
(62% del CSV) son la fase "off" de un pulso (PWM armado pero sin orden de giro),
no frenos. Por eso el dataset tiene 4 clases de direccion y ninguna es Stop.

AGREGACION POR VENTANA
----------------------
El control esta PULSADO (adelante/off/adelante/off a varios Hz con el PWM fijo).
Para cada imagen tomamos las filas del CSV en una ventana CENTRADA
[t - W/2, t + W/2) y calculamos la accion DOMINANTE ponderada por el tiempo que
duro cada estado. Si la ventana cae entera en una racha de "off" (sin ninguna
direccion activa), se usa la fila ACTIVA mas cercana en el tiempo.

Hardware (L298N):
  - speedA/speedB = PWM (0-100%) al enable del motor IZQ / DER  -> consigna, no sensor
  - motor IZQ: (GPIO1,GPIO2) = (1,0) adelante ; (0,1) atras ; (0,0) sin orden
  - motor DER: (GPIO3,GPIO4) = (0,1) adelante ; (1,0) atras ; (0,0) sin orden
"""

import argparse
import glob
import os
from bisect import bisect_left

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Estado instantaneo de cada motor
# --------------------------------------------------------------------------- #
def left_state(speed, g1, g2):
    """Motor izquierdo: 'fwd', 'bwd' o 'stop'."""
    if speed <= 0:
        return "stop"
    if g1 == 1 and g2 == 0:
        return "fwd"
    if g1 == 0 and g2 == 1:
        return "bwd"
    return "stop"  # (0,0) con speed>0 = fase off de un pulso


def right_state(speed, g3, g4):
    """Motor derecho: 'fwd', 'bwd' o 'stop' (convencion invertida)."""
    if speed <= 0:
        return "stop"
    if g3 == 0 and g4 == 1:
        return "fwd"
    if g3 == 1 and g4 == 0:
        return "bwd"
    return "stop"


def classify(net_l, net_r, turn_th):
    """
    Mapea el balance de los motores a una de las 4 direcciones (sin Stop).
      net_l, net_r in [-1, 1]: fraccion de tiempo neta hacia adelante (signo).
      diff > 0  => el motor izquierdo empuja mas  => el auto gira a la DERECHA.
    """
    avg = (net_l + net_r) / 2.0
    diff = net_l - net_r
    if avg >= 0:  # familia hacia adelante
        if abs(diff) <= turn_th:
            return "Forward"
        return "Right" if diff > 0 else "Left"
    else:         # familia hacia atras
        if abs(diff) <= turn_th:
            return "Backward"
        return "Right" if diff > 0 else "Left"


def speed_to_level(value, low_max, high_min):
    """Bucketiza la velocidad efectiva (PWM) en Low / Medium / High."""
    if value <= low_max:
        return "Low"
    if value >= high_min:
        return "High"
    return "Medium"


# --------------------------------------------------------------------------- #
# Agregacion de una ventana de filas de control
# --------------------------------------------------------------------------- #
def aggregate_window(rows, win_start, win_end):
    """
    rows: lista de (t, speedA, speedB, g1, g2, g3, g4) dentro de la ventana,
          ordenadas por t. Devuelve (net_l, net_r, speed_value, has_active).
    Pondera cada fila por la duracion que estuvo vigente.
    """
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0.0, False

    t_fwd_l = t_bwd_l = t_fwd_r = t_bwd_r = 0.0
    total = 0.0
    speed_sum = speed_dur = 0.0

    for k in range(n):
        t, sa, sb, g1, g2, g3, g4 = rows[k]
        end = rows[k + 1][0] if k + 1 < n else win_end
        dur = max(end - t, 1.0)
        total += dur

        ls = left_state(sa, g1, g2)
        rs = right_state(sb, g3, g4)
        if ls == "fwd":
            t_fwd_l += dur
        elif ls == "bwd":
            t_bwd_l += dur
        if rs == "fwd":
            t_fwd_r += dur
        elif rs == "bwd":
            t_bwd_r += dur

        if ls != "stop":
            speed_sum += sa * dur
            speed_dur += dur
        if rs != "stop":
            speed_sum += sb * dur
            speed_dur += dur

    net_l = (t_fwd_l - t_bwd_l) / total
    net_r = (t_fwd_r - t_bwd_r) / total
    has_active = speed_dur > 0
    speed_value = speed_sum / speed_dur if has_active else 0.0
    return net_l, net_r, speed_value, has_active


# --------------------------------------------------------------------------- #
# Procesamiento de una sesion (#Record_*)
# --------------------------------------------------------------------------- #
def process_record(record_dir, dataset_dir, window_ms, turn_th, low_max, high_min):
    record = os.path.basename(record_dir)
    images_dir = os.path.join(record_dir, "Images")
    csv_path = os.path.join(record_dir, "labels.csv")

    if not os.path.isdir(images_dir) or not os.path.isfile(csv_path):
        print(f"  [!] {record}: falta Images/ o labels.csv, se omite")
        return []

    # --- control ordenado por tiempo ---
    df = pd.read_csv(csv_path, sep=";").sort_values("time_in_ms").reset_index(drop=True)
    ctrl = list(zip(
        df["time_in_ms"].astype(float),
        df["speedA"].astype(float), df["speedB"].astype(float),
        df["GPIO1"].astype(int), df["GPIO2"].astype(int),
        df["GPIO3"].astype(int), df["GPIO4"].astype(int),
    ))
    ctrl_times = [r[0] for r in ctrl]

    # --- filas ACTIVAS (al menos un motor girando) para el fallback ---
    active = []  # (t, direction, speed)
    for t, sa, sb, g1, g2, g3, g4 in ctrl:
        ls, rs = left_state(sa, g1, g2), right_state(sb, g3, g4)
        l = 1 if ls == "fwd" else (-1 if ls == "bwd" else 0)
        r = 1 if rs == "fwd" else (-1 if rs == "bwd" else 0)
        if l == 0 and r == 0:
            continue  # fila de pulso-off, no aporta direccion
        spd = (sa if l else 0) + (sb if r else 0)
        cnt = (1 if l else 0) + (1 if r else 0)
        active.append((t, classify(l, r, turn_th), spd / cnt))
    active_times = [a[0] for a in active]

    def nearest_active(t):
        """Direccion y velocidad de la fila activa mas cercana en el tiempo."""
        if not active:
            return "Forward", 50.0  # no deberia pasar
        idx = bisect_left(active_times, t)
        cand = [j for j in (idx, idx - 1) if 0 <= j < len(active)]
        j = min(cand, key=lambda k: abs(active_times[k] - t))
        return active[j][1], active[j][2]

    # --- imagenes ordenadas por timestamp ---
    imgs = []
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(".png"):
            try:
                ts = int(os.path.splitext(fn)[0])
            except ValueError:
                continue
            imgs.append((ts, fn))
    imgs.sort()
    if not imgs:
        return []

    half = window_ms / 2.0
    rows_out = []
    for ts, fn in imgs:
        win_start, win_end = ts - half, ts + half
        lo = bisect_left(ctrl_times, win_start)
        hi = bisect_left(ctrl_times, win_end)
        window_rows = ctrl[lo:hi]

        net_l, net_r, speed_value, has_active = aggregate_window(
            window_rows, win_start, win_end
        )
        if has_active:
            direction = classify(net_l, net_r, turn_th)
        else:
            # ventana vacia o solo pulso-off -> fila activa mas cercana
            direction, speed_value = nearest_active(ts)

        rows_out.append({
            "image_path": os.path.join(dataset_dir, record, "Images", fn),
            "direction": direction,
            "speed_level": speed_to_level(speed_value, low_max, high_min),
            "record": record,
            "time_in_ms": ts,
            "n_control": len(window_rows),
        })
    return rows_out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Construye el dataset maestro imagen->accion (4 clases, sin Stop).")
    ap.add_argument("--dataset-dir", default="DataSet", help="carpeta raiz con los #Record_*")
    ap.add_argument("--output", default="dataset.csv", help="archivo de salida")
    ap.add_argument("--window-ms", type=int, default=700,
                    help="ancho de la ventana CENTRADA en ms (default 700, ~1 ciclo de pulso)")
    ap.add_argument("--turn-threshold", type=float, default=0.20,
                    help="diferencia entre motores a partir de la cual es giro (vs recto)")
    ap.add_argument("--low-max", type=float, default=45.0, help="PWM <= este valor => Low")
    ap.add_argument("--high-min", type=float, default=55.0, help="PWM >= este valor => High")
    args = ap.parse_args()

    record_dirs = sorted(
        d for d in glob.glob(os.path.join(args.dataset_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d).startswith("#Record")
    )
    if not record_dirs:
        print(f"No se encontraron carpetas #Record_* en '{args.dataset_dir}'.")
        return

    print(f"Sesiones encontradas: {len(record_dirs)}")
    all_rows = []
    for rd in record_dirs:
        rows = process_record(
            rd, args.dataset_dir, args.window_ms,
            args.turn_threshold, args.low_max, args.high_min,
        )
        print(f"  {os.path.basename(rd):32s} -> {len(rows):5d} imagenes etiquetadas")
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows, columns=[
        "image_path", "direction", "speed_level", "record", "time_in_ms", "n_control"
    ])
    out.to_csv(args.output, index=False)

    print("\n" + "=" * 60)
    print(f"DATASET MAESTRO: {len(out)} filas -> {args.output}")
    print("=" * 60)
    print("\n-- Distribucion DIRECTION (%) --")
    print((out["direction"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\n-- Distribucion SPEED_LEVEL (%) --")
    print((out["speed_level"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\n-- DIRECTION x SPEED_LEVEL (conteo) --")
    print(pd.crosstab(out["direction"], out["speed_level"]).to_string())
    print("\n-- Filas por sesion --")
    print(out["record"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

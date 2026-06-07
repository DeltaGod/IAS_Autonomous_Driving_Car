"""
build_global_csv.py
===================
Genera un CSV global con UNA fila por imagen de todas las sesiones.

Columnas: las del labels.csv original + behavior, record y source:
    time_in_ms , speedA , speedB , GPIO1 , GPIO2 , GPIO3 , GPIO4 , behavior , record , source

  - time_in_ms : timestamp de la IMAGEN (= nombre del archivo <ts>.png)
  - behavior   : STOP / FORWARD / BACKWARD / LEFT / RIGHT
  - record     : carpeta de sesion de la imagen
  - source     : 'real' (comando real asociado) o 'interp' (comando interpolado)

LOGICA
------
1) CORRECCION DE ERROR (freno): el driver L298N frena el motor si los 4 GPIO de
   sentido estan en 0. Si una fila tiene los 4 GPIO en 0 y alguna velocidad != 0,
   es un error: se fuerza speedA = speedB = 0.

2) ASOCIACION: cada comando se asocia a la imagen ANTERIOR mas cercana (ultima
   imagen con timestamp <= tiempo del comando). Esa imagen toma los valores
   exactos del comando.

3) RECORTE DE BORDES: el piloto dejaba el auto quieto largos periodos ANTES del
   primer comando y DESPUES del ultimo. Esas imagenes se DESCARTAN.

4) INTERPOLACION: las imagenes sin comando asociado reciben velocidades interpoladas.
   Los GPIO reciben la DIRECCION del lado que se esta moviendo.

5) GPIO EN INTERPOLADAS: si la velocidad interpolada es > 0, se mantiene la direccion.
   Solo si la velocidad interpolada es 0 sus GPIO van a (0,0).

6) DECODIFICAR BEHAVIOR SIN UMBRAL: direccion decidida con velocidad + GPIO.

7) FILTRO DE LATENCIA EXCESIVA (NUEVO): Si la diferencia de tiempo entre dos frames
   consecutivos es mayor a 1000ms, se descarta el frame para evitar alucinaciones
   por saltos temporales/teletransportes y se generan en el csv distintas "sesiones" de trabajo
   (para respetar los tiempos de guardado originales, sino habría que meter un offset)
"""

import argparse
import glob
import os
from bisect import bisect_left, bisect_right

import pandas as pd

MAX_DELAY_MS = 1000  # Umbral máximo para considerar que hubo un salto/teletransporte

def apply_brake_rule(sa, sb, g1, g2, g3, g4):
    """Si los 4 GPIO (redondeados) estan en 0 -> motor frenado -> velocidades a 0."""
    if round(g1) == 0 and round(g2) == 0 and round(g3) == 0 and round(g4) == 0:
        return 0.0, 0.0
    return sa, sb

def decode_behavior(sa, sb, g1, g2, g3, g4):
    """Comportamiento a partir de velocidad + GPIO, SIN umbral."""
    G1, G2, G3, G4 = (int(round(g1)), int(round(g2)), int(round(g3)), int(round(g4)))
    l = 0 if sa <= 0 else (1 if (G1, G2) == (1, 0) else (-1 if (G1, G2) == (0, 1) else 0))
    r = 0 if sb <= 0 else (1 if (G3, G4) == (0, 1) else (-1 if (G3, G4) == (1, 0) else 0))
    if l == 0 and r == 0:
        return "STOP"
    if l > 0 and r > 0:
        return "FORWARD"
    if l < 0 and r < 0:
        return "BACKWARD"
    return "RIGHT" if (l - r) > 0 else "LEFT"

def pick_dir(speed_i, sa_a, dir_a, sa_b, dir_b):
    """Direccion (par de GPIO) de un motor en una imagen INTERPOLADA."""
    if speed_i <= 0:
        return (0, 0)
    if sa_a > 0 and dir_a != (0, 0):
        return dir_a
    if sa_b > 0 and dir_b != (0, 0):
        return dir_b
    return (0, 0)

def interp_fields(t, cmds, cmd_times):
    """Interpola los campos del comando en el tiempo t para una imagen sin comando."""
    hi = bisect_left(cmd_times, t)
    if hi == 0:
        c = cmds[0]
        return (round(c[1], 1), round(c[2], 1), c[3], c[4], c[5], c[6])
    if hi >= len(cmds):
        c = cmds[-1]
        return (round(c[1], 1), round(c[2], 1), c[3], c[4], c[5], c[6])

    a, b = cmds[hi - 1], cmds[hi]
    ta, tb = a[0], b[0]
    w = 0.0 if tb == ta else (t - ta) / (tb - ta)
    sa = round(a[1] + (b[1] - a[1]) * w, 1)
    sb = round(a[2] + (b[2] - a[2]) * w, 1)
    g1, g2 = pick_dir(sa, a[1], (a[3], a[4]), b[1], (b[3], b[4]))
    g3, g4 = pick_dir(sb, a[2], (a[5], a[6]), b[2], (b[5], b[6]))
    return (sa, sb, g1, g2, g3, g4)

def process_record(record_dir, dataset_dir):
    record = os.path.basename(record_dir)
    images_dir = os.path.join(record_dir, "Images")
    csv_path = os.path.join(record_dir, "labels.csv")
    if not os.path.isdir(images_dir) or not os.path.isfile(csv_path):
        return [], 0, 0, 0, 0

    df = pd.read_csv(csv_path, sep=";").sort_values("time_in_ms").reset_index(drop=True)
    cmds = []
    for _, row in df.iterrows():
        sa, sb = float(row.speedA), float(row.speedB)
        g1, g2, g3, g4 = int(row.GPIO1), int(row.GPIO2), int(row.GPIO3), int(row.GPIO4)
        sa, sb = apply_brake_rule(sa, sb, g1, g2, g3, g4)
        cmds.append((float(row.time_in_ms), sa, sb, g1, g2, g3, g4))
    cmd_times = [c[0] for c in cmds]

    imgs = []
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(".png"):
            try:
                ts = int(os.path.splitext(fn)[0])
            except ValueError:
                continue
            imgs.append(ts)
    imgs.sort()
    
    if not imgs or not cmds:
        return [], 0, 0, 0, 0

    direct = {}
    for c in cmds:
        j = bisect_right(imgs, c[0]) - 1
        if j >= 0:
            direct[j] = c[1:]
    
    if not direct:
        return [], 0, 0, 0, 0

    first_idx = min(direct)
    last_idx = max(direct)
    n_dropped_bordes = first_idx + (len(imgs) - 1 - last_idx)

    rows = []
    n_cmd = n_interp = n_dropped_delay = 0
    last_ts = None
    chunk_id = 1
    current_record = f"{record}_seq{chunk_id}"

    for i in range(first_idx, last_idx + 1):
        ts = imgs[i]

        # --- FILTRO DE LATENCIA Y SEGMENTACIÓN DE SESIONES ---
        if last_ts is not None and (ts - last_ts) > MAX_DELAY_MS:
            n_dropped_delay += 1
            chunk_id += 1
            current_record = f"{record}_seq{chunk_id}" # Cortamos la serie temporal aquí
            last_ts = ts # Seteamos la nueva base de tiempo para el siguiente frame
            continue # Eliminamos el frame de transición

        last_ts = ts

        if i in direct:
            sa, sb, g1, g2, g3, g4 = direct[i]
            source = "real"
            n_cmd += 1
        else:
            sa, sb, g1, g2, g3, g4 = interp_fields(ts, cmds, cmd_times)
            source = "interp"
            n_interp += 1

        sa, sb = apply_brake_rule(sa, sb, g1, g2, g3, g4)
        rows.append({
            "image_path": os.path.join(dataset_dir, record, "Images", f"{ts}.png"),
            "time_in_ms": ts,
            "speedA": round(sa, 1),
            "speedB": round(sb, 1),
            "GPIO1": int(round(g1)),
            "GPIO2": int(round(g2)),
            "GPIO3": int(round(g3)),
            "GPIO4": int(round(g4)),
            "behavior": decode_behavior(sa, sb, g1, g2, g3, g4),
            "record": current_record, # Inyectamos el nombre de sesión segmentado
            "source": source,
        })
    return rows, n_cmd, n_interp, n_dropped_bordes, n_dropped_delay


def main():
    ap = argparse.ArgumentParser(description="Genera un CSV global imagen->comando con segmentación temporal.")
    ap.add_argument("--dataset-dir", default="DataSet", help="carpeta raiz con los Record_*")
    ap.add_argument("--output", default="dataset_global.csv", help="archivo de salida")
    args = ap.parse_args()

    record_dirs = sorted(
        d for d in glob.glob(os.path.join(args.dataset_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d).lstrip("#").startswith("Record")
    )
    if not record_dirs:
        print(f"No se encontraron carpetas Record_* en '{args.dataset_dir}'.")
        return

    print(f"Sesiones encontradas: {len(record_dirs)}")
    all_rows = []
    total_dropped_bordes = 0
    total_dropped_delay = 0
    
    for rd in record_dirs:
        rows, n_cmd, n_interp, n_drop_bordes, n_drop_delay = process_record(rd, args.dataset_dir)
        total_dropped_bordes += n_drop_bordes
        total_dropped_delay += n_drop_delay
        print(f"  {os.path.basename(rd):30s} -> {len(rows):6d} imgs  "
              f"(cmd: {n_cmd}, interp: {n_interp}, bordes desc.: {n_drop_bordes}, delays desc.: {n_drop_delay})")
        all_rows.extend(rows)

    cols = ["image_path", "time_in_ms", "speedA", "speedB", "GPIO1", "GPIO2", "GPIO3", "GPIO4",
            "behavior", "record", "source"]
    out = pd.DataFrame(all_rows, columns=cols)
    out.to_csv(args.output, index=False)

    print("\n" + "=" * 60)
    print(f"CSV GLOBAL: {len(out)} filas -> {args.output}")
    print(f"Imágenes descartadas en bordes (inactividad): {total_dropped_bordes}")
    print(f"Imágenes descartadas por saltos (>1s): {total_dropped_delay}")
    print("=" * 60)
    print("\n-- Distribución BEHAVIOR (%) --")
    print((out["behavior"].value_counts(normalize=True) * 100).round(1).to_string())

if __name__ == "__main__":
    main()
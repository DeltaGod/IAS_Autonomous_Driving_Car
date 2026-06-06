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

3) INTERPOLACION: las imagenes sin comando asociado reciben los 6 campos
   interpolados linealmente por tiempo entre el comando real anterior y el
   posterior (clamp en los bordes). GPIO se redondea a {0,1} (lineas binarias).

SIN UMBRAL: un motor se considera parado solo si su velocidad es 0 (no hay corte
arbitrario). En los giros pivote un motor queda en velocidad 0 mientras el otro
sigue, asi que la direccion se decide con velocidad + GPIO.

Convencion de motores (L298N):
  - motor IZQ: (GPIO1,GPIO2) = (1,0) adelante ; (0,1) atras
  - motor DER: (GPIO3,GPIO4) = (0,1) adelante ; (1,0) atras
"""

import argparse
import glob
import os
from bisect import bisect_left, bisect_right

import pandas as pd


def apply_brake_rule(sa, sb, g1, g2, g3, g4):
    """Si los 4 GPIO (redondeados) estan en 0 -> motor frenado -> velocidades a 0."""
    if round(g1) == 0 and round(g2) == 0 and round(g3) == 0 and round(g4) == 0:
        return 0.0, 0.0
    return sa, sb


def decode_behavior(sa, sb, g1, g2, g3, g4):
    """
    Comportamiento a partir de velocidad + GPIO, SIN umbral.
    Un motor con velocidad 0 esta parado aunque su GPIO marque un sentido
    (asi se hacen los giros pivote en esta data).
    """
    G1, G2, G3, G4 = (int(round(g1)), int(round(g2)), int(round(g3)), int(round(g4)))
    l = 0 if sa <= 0 else (1 if (G1, G2) == (1, 0) else (-1 if (G1, G2) == (0, 1) else 0))
    r = 0 if sb <= 0 else (1 if (G3, G4) == (0, 1) else (-1 if (G3, G4) == (1, 0) else 0))
    if l == 0 and r == 0:
        return "STOP"
    if l > 0 and r > 0:
        return "FORWARD"
    if l < 0 and r < 0:
        return "BACKWARD"
    # diff > 0 => motor izq empuja mas => gira a la derecha
    return "RIGHT" if (l - r) > 0 else "LEFT"


def interpolate(t, cmds, cmd_times):
    """Interpola los 6 campos del comando en el tiempo t entre los comandos que lo bracketean."""
    hi = bisect_left(cmd_times, t)  # primer comando con tiempo >= t
    if hi == 0:
        return cmds[0][1:]          # antes del primer comando: clamp
    if hi >= len(cmds):
        return cmds[-1][1:]         # despues del ultimo comando: clamp
    a, b = cmds[hi - 1], cmds[hi]
    ta, tb = a[0], b[0]
    if tb == ta:
        return b[1:]
    w = (t - ta) / (tb - ta)
    return tuple(a[1 + k] + (b[1 + k] - a[1 + k]) * w for k in range(6))


def process_record(record_dir, dataset_dir):
    record = os.path.basename(record_dir)
    images_dir = os.path.join(record_dir, "Images")
    csv_path = os.path.join(record_dir, "labels.csv")
    if not os.path.isdir(images_dir) or not os.path.isfile(csv_path):
        print(f"  [!] {record}: falta Images/ o labels.csv, se omite")
        return [], 0, 0

    # --- comandos: leer, ordenar, aplicar correccion de freno ---
    df = pd.read_csv(csv_path, sep=";").sort_values("time_in_ms").reset_index(drop=True)
    cmds = []
    for _, row in df.iterrows():
        sa, sb = float(row.speedA), float(row.speedB)
        g1, g2, g3, g4 = int(row.GPIO1), int(row.GPIO2), int(row.GPIO3), int(row.GPIO4)
        sa, sb = apply_brake_rule(sa, sb, g1, g2, g3, g4)
        cmds.append((float(row.time_in_ms), sa, sb, g1, g2, g3, g4))
    cmd_times = [c[0] for c in cmds]

    # --- imagenes ordenadas por timestamp ---
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
        return [], 0, 0

    # --- asociacion directa: cada comando -> imagen anterior mas cercana ---
    direct = {}  # indice de imagen -> campos del comando (ultimo gana si colisionan)
    for c in cmds:
        j = bisect_right(imgs, c[0]) - 1  # ultima imagen con ts <= tiempo del comando
        if j >= 0:
            direct[j] = c[1:]

    # --- una fila por imagen ---
    rows = []
    n_cmd = n_interp = 0
    for i, ts in enumerate(imgs):
        if i in direct:
            fields = direct[i]
            source = "real"
            n_cmd += 1
        else:
            fields = interpolate(ts, cmds, cmd_times)
            source = "interp"
            n_interp += 1

        sa, sb, g1, g2, g3, g4 = fields
        sa, sb = apply_brake_rule(sa, sb, g1, g2, g3, g4)  # re-aplicar freno tras interpolar
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
            "record": record,
            "source": source,
        })
    return rows, n_cmd, n_interp


def main():
    ap = argparse.ArgumentParser(description="Genera un CSV global imagen->comando con columna behavior.")
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
    for rd in record_dirs:
        rows, n_cmd, n_interp = process_record(rd, args.dataset_dir)
        print(f"  {os.path.basename(rd):30s} -> {len(rows):6d} imgs  (cmd: {n_cmd}, interp: {n_interp})")
        all_rows.extend(rows)

    cols = ["image_path", "time_in_ms", "speedA", "speedB", "GPIO1", "GPIO2", "GPIO3", "GPIO4",
            "behavior", "record", "source"]
    out = pd.DataFrame(all_rows, columns=cols)
    out.to_csv(args.output, index=False)

    print("\n" + "=" * 60)
    print(f"CSV GLOBAL: {len(out)} filas -> {args.output}")
    print("=" * 60)
    print("\n-- Distribucion BEHAVIOR (%) --")
    print((out["behavior"].value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == "__main__":
    main()

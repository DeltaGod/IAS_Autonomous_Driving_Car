"""
build_signed_csv.py
===================
Genera las bases SIGNED (train/val) para el experimento de VELOCIDAD CON SIGNO.

En vez de (dirección {STOP,FWD,BWD} + velocidad [0,100]) por motor, el target es UNA
variable continua por motor: el PWM con SIGNO ∈ [-100, 100]:
    > 0  -> FORWARD     < 0  -> BACKWARD     = 0  -> STOP

POR QUÉ ENVUELVE build_global_csv (y no reimplementa):
toda la lógica delicada (regla de freno, asociación comando->imagen anterior, recorte
de bordes, filtro de latencia >1s con segmentación en _seqN, interpolación + pick_dir,
decode por motor) YA está validada ahí. Acá la reusamos tal cual y solo TRANSFORMAMOS
el target al final:
    signed_speedA = speedA * signo(behaviorA)   (FWD=+1, BWD=-1, STOP=0)
    signed_speedB = speedB * signo(behaviorB)
Como el decode garantiza STOP <=> velocidad 0, |signed| == speed y el signo es la
dirección. No se interpola "a través del cero" (eso sería reimplementar la interpolación);
la diferencia solo afecta reversas FWD->BWD sin STOP intermedio, casi inexistentes acá.

COLUMNAS DE SALIDA:
    image_path, signed_speedA, signed_speedB,   <- lo que CONSUME el modelo (imagen->2 continuas)
    behaviorA, behaviorB, behavior,             <- debug + flip L<->R / resample por clase
    time_in_ms, record, source                  <- trazabilidad

NO toca dataset_train.csv / dataset_val.csv (las viejas siguen intactas).
"""

import argparse
import glob
import os

import pandas as pd

# Reusamos TODA la maquinaria validada del builder original.
from build_global_csv import build_split, _DIR_SIGN

SIGNED_COLS = [
    "image_path", "signed_speedA", "signed_speedB",   # consume el modelo
    "behaviorA", "behaviorB", "behavior",             # debug + flip/resample
    "time_in_ms", "record", "source",                 # trazabilidad
]


def to_signed_rows(rows):
    """Transforma las filas del builder original (speed+dir+GPIO) a target con signo."""
    out = []
    for r in rows:
        sa = round(r["speedA"] * _DIR_SIGN[r["behaviorA"]], 1)
        sb = round(r["speedB"] * _DIR_SIGN[r["behaviorB"]], 1)
        out.append({
            "image_path": r["image_path"],
            "signed_speedA": sa,
            "signed_speedB": sb,
            "behaviorA": r["behaviorA"],
            "behaviorB": r["behaviorB"],
            "behavior": r["behavior"],
            "time_in_ms": r["time_in_ms"],
            "record": r["record"],
            "source": r["source"],
        })
    return out


def dump_split(rows, path, name):
    out = pd.DataFrame(rows, columns=SIGNED_COLS)
    out.to_csv(path, index=False)
    print("\n" + "=" * 60)
    print(f"CSV SIGNED {name}: {len(out)} filas -> {path}")
    print("=" * 60)
    if not len(out):
        return
    # Verificación: distribución del signo por motor (debe espejar la dirección).
    for col in ["signed_speedA", "signed_speedB"]:
        v = out[col]
        z = (v == 0).mean() * 100
        p = (v > 0).mean() * 100
        n = (v < 0).mean() * 100
        mag = v[v != 0].abs()
        print(f"  {col:14s}: STOP(0) {z:5.1f}% | FWD(+) {p:5.1f}% | BWD(-) {n:5.1f}%"
              f"  | |v|≠0: min {mag.min():.0f} mean {mag.mean():.1f} max {mag.max():.0f}")
    print(f"  source: {dict(out['source'].value_counts())}")
    print(f"  -- behavior global (%) --")
    print((out["behavior"].value_counts(normalize=True) * 100).round(1).to_string())


def main():
    ap = argparse.ArgumentParser(description="Genera CSVs SIGNED (imagen->PWM con signo por motor).")
    ap.add_argument("--dataset-dir", default="DataSet", help="carpeta raiz con los Record_*")
    ap.add_argument("--output-train", default="dataset_train_signed.csv", help="CSV de entrenamiento SIGNED")
    ap.add_argument("--output-val", default="dataset_val_signed.csv", help="CSV de validación SIGNED")
    args = ap.parse_args()

    record_dirs = sorted(
        d for d in glob.glob(os.path.join(args.dataset_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d).lstrip("#").startswith("Record")
    )
    if not record_dirs:
        print(f"No se encontraron carpetas Record_* en '{args.dataset_dir}'.")
        return

    # MISMO split por sesión que build_global_csv: 1ª sesión (la grande) -> TRAIN, resto -> VAL.
    train_dirs = record_dirs[:1]
    val_dirs = record_dirs[1:]
    print(f"Sesiones: {len(record_dirs)}")
    print(f"  TRAIN: {', '.join(os.path.basename(d) for d in train_dirs)}")
    print(f"  VAL  : {', '.join(os.path.basename(d) for d in val_dirs) or '(ninguna)'}")

    print("\n[TRAIN]")
    train_rows, _, _ = build_split(train_dirs, args.dataset_dir)
    print("\n[VAL]")
    val_rows, _, _ = build_split(val_dirs, args.dataset_dir)

    dump_split(to_signed_rows(train_rows), args.output_train, "TRAIN")
    dump_split(to_signed_rows(val_rows), args.output_val, "VAL")


if __name__ == "__main__":
    main()

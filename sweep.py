"""
sweep.py — Exploración de hiperparámetros GENÉRICA (frontera de Pareto + anti-ruido).
=====================================================================================
Barre una grilla de hiperparámetros sobre CUALQUIER modelo (hybrid / perframe / signed /
sequence), reusando su `run_experiment`. No pisa los resultados viejos: todo va a
`results/sweeps/<nombre>/`. Cada config corre en un SUBPROCESO aislado (evita que la
memoria CUDA se acumule entre corridas en la RAM unificada de la Jetson).

QUÉ PRODUCE (en results/sweeps/<nombre>/):
  - <NN>_<tag>/            : carpeta de resultados de cada config (eval, report, metrics, ckpt)
  - sweep_summary.csv      : una fila por config con sus hiperparámetros + métricas del MEJOR
                             checkpoint (el que elige el modelo: máx val_dir_f1_macro).
  - pareto.png             : scatter F1-macro vs full-stop-recall, con la FRONTERA DE PARETO
                             marcada (las configs no dominadas: nadie les gana en ambos ejes).
  - param_effects.png      : efecto marginal de cada parámetro sobre F1 y full-stop.
  - reseed_summary.csv     : ANTI-RUIDO. Re-corre los finalistas (los de Pareto) con varias
                             semillas y reporta media ± desvío -> ¿la ventaja sobrevive al
                             ruido del val (±0.05) o era suerte?

USO:
  python3 sweep.py                      # corre el sweep definido en SWEEP (abajo)
  python3 sweep.py --run-one ...        # (interno) corre UNA config en subproceso

La grilla y el modelo objetivo se editan en el dict SWEEP, al final del archivo.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

# nombre de motor -> (módulo, función run_experiment)
ENGINES = {
    "hybrid": ("nn_hybrid", "run_experiment"),
    "perframe": ("nn_perframe", "run_experiment"),
    "signed": ("nn_signed", "run_experiment"),
    "sequence": ("nn_sequence", "run_experiment"),
}


# =====================================================================
# WORKER: corre UNA config (en subproceso aislado)
# =====================================================================
def run_one(engine, tag, results_dir, seed, cfg_json):
    import importlib
    import pytorch_lightning as pl
    pl.seed_everything(int(seed), workers=True)   # semilla global: init, shuffle, augment
    mod, fn = ENGINES[engine]
    run = getattr(importlib.import_module(mod), fn)
    cfg = replace(Config(**json.loads(cfg_json)), seed=int(seed))
    run(cfg, tag, results_dir=results_dir)


# =====================================================================
# COLECCIÓN de métricas
# =====================================================================
def find_metrics_csv(run_dir):
    logs = os.path.join(run_dir, "logs")
    if not os.path.isdir(logs):
        return None
    for v in sorted(os.listdir(logs)):
        p = os.path.join(logs, v, "metrics.csv")
        if os.path.exists(p):
            return p
    return None


def best_operating_point(metrics_csv, select="val_dir_f1_macro", mode="max"):
    """Métricas en la época del MEJOR checkpoint (el criterio con que el modelo elige)."""
    if not metrics_csv or not os.path.exists(metrics_csv):
        return None
    m = pd.read_csv(metrics_csv)
    g = m.groupby("epoch").max(numeric_only=True)   # junta filas train/val de cada época
    if select not in g or g[select].dropna().empty:
        return None
    idx = g[select].idxmax() if mode == "max" else g[select].idxmin()
    return g.loc[idx].to_dict()


def pareto_mask(xs, ys):
    """True si el punto NO está dominado (ambos ejes a MAXIMIZAR)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = len(xs); keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                keep[i] = False
                break
    return keep


# =====================================================================
# ORQUESTADOR
# =====================================================================
def _short(k, v):
    key = "".join(w[0] for w in k.split("_"))                 # lambda_stop -> ls
    sv = str(v).replace(" ", "").replace("[", "").replace("]", "").replace(",", "-")
    return f"{key}{sv}"


def launch(engine, tag, results_dir, seed, cfg):
    """Corre una config en subproceso; devuelve (returncode, métricas-del-mejor-ckpt)."""
    cfg_json = json.dumps(asdict(cfg))
    log = os.path.join(results_dir, tag + ".log")
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--run-one",
           "--engine", engine, "--tag", tag, "--results-dir", results_dir,
           "--seed", str(seed), "--cfg", cfg_json]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    op = best_operating_point(find_metrics_csv(os.path.join(results_dir, tag)))
    return rc, (op or {})


METRIC_COLS = ["val_dir_f1_macro", "val_fullstop_recall",
               "val_signed_mae_avg", "val_signed_mae_A", "val_signed_mae_B",
               "val_mae_A", "val_mae_B"]


def run_sweep(name, engine, base, grid, axis_x="val_dir_f1_macro",
              axis_y="val_fullstop_recall", reseed_seeds=(101, 202)):
    results_dir = os.path.join("results", "sweeps", name)
    os.makedirs(results_dir, exist_ok=True)
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    print(f"[sweep {name}] engine={engine} | {len(combos)} configs | ejes Pareto: {axis_x} vs {axis_y}",
          flush=True)

    records = []
    for i, ov in enumerate(combos):
        tag = f"{i:02d}_" + "_".join(_short(k, ov[k]) for k in keys)
        cfg = replace(base, **ov)
        t0 = time.time()
        print(f"[sweep] ({i+1}/{len(combos)}) {tag} ...", flush=True)
        rc, op = launch(engine, tag, results_dir, base.seed, cfg)
        rec = {"tag": tag, "seed": base.seed, "rc": rc, "secs": round(time.time() - t0)}
        rec.update({k: ov[k] for k in keys})
        rec.update({c: op.get(c) for c in METRIC_COLS if c in op})
        rec["_override"] = ov
        records.append(rec)
        _dump_summary(records, keys, os.path.join(results_dir, "sweep_summary.csv"))
        print(f"    -> rc={rc} {axis_x}={op.get(axis_x)} {axis_y}={op.get(axis_y)} ({rec['secs']}s)",
              flush=True)

    df = _dump_summary(records, keys, os.path.join(results_dir, "sweep_summary.csv"))
    ok = df.dropna(subset=[axis_x, axis_y]).reset_index(drop=True)
    if ok.empty:
        print("[sweep] ninguna config produjo métricas; abortando plots."); return
    ok = ok.copy()
    ok["pareto"] = pareto_mask(ok[axis_x], ok[axis_y])
    _dump_summary(records, keys, os.path.join(results_dir, "sweep_summary.csv"), pareto_df=ok)
    plot_pareto(ok, axis_x, axis_y, os.path.join(results_dir, "pareto.png"), name)
    plot_param_effects(ok, keys, axis_x, axis_y, os.path.join(results_dir, "param_effects.png"))

    # ---------- ANTI-RUIDO: reseed de los finalistas (frontera de Pareto) ----------
    finalists = [records[i] for i in ok.index[ok["pareto"]].tolist()]
    print(f"[sweep] reseed anti-ruido de {len(finalists)} finalistas con seeds {reseed_seeds}", flush=True)
    rs = []
    for rec in finalists:
        ov = rec["_override"]
        vals = {axis_x: [rec.get(axis_x)], axis_y: [rec.get(axis_y)]}
        for s in reseed_seeds:
            tag = rec["tag"] + f"_seed{s}"
            _, op = launch(engine, tag, results_dir, s, replace(base, **ov, seed=s))
            vals[axis_x].append(op.get(axis_x)); vals[axis_y].append(op.get(axis_y))
        row = {"tag": rec["tag"], "n_seeds": 1 + len(reseed_seeds)}
        for ax in (axis_x, axis_y):
            v = np.array([x for x in vals[ax] if x is not None], float)
            row[f"{ax}_mean"] = round(float(v.mean()), 4) if len(v) else None
            row[f"{ax}_std"] = round(float(v.std()), 4) if len(v) else None
        rs.append(row)
        pd.DataFrame(rs).to_csv(os.path.join(results_dir, "reseed_summary.csv"), index=False)
        print(f"    {row['tag']}: {axis_x}={row.get(axis_x+'_mean')}±{row.get(axis_x+'_std')} "
              f"{axis_y}={row.get(axis_y+'_mean')}±{row.get(axis_y+'_std')}", flush=True)
    print(f"\n[sweep {name}] LISTO. Resumen en {results_dir}/", flush=True)


def _dump_summary(records, keys, path, pareto_df=None):
    cols = ["tag", "seed", "rc", "secs"] + keys + [c for c in METRIC_COLS
            if any(c in r for r in records)]
    df = pd.DataFrame([{k: r.get(k) for k in cols} for r in records])
    if pareto_df is not None:
        df = df.merge(pareto_df[["tag", "pareto"]], on="tag", how="left")
    df.to_csv(path, index=False)
    return df


# =====================================================================
# GRÁFICOS
# =====================================================================
def plot_pareto(df, ax_x, ax_y, out_png, name):
    fig, ax = plt.subplots(figsize=(9, 7))
    dom = df[~df["pareto"]]; par = df[df["pareto"]].sort_values(ax_x)
    ax.scatter(dom[ax_x], dom[ax_y], c="#9aa0a6", s=45, label="dominadas", zorder=2)
    ax.scatter(par[ax_x], par[ax_y], c="#d62728", s=90, edgecolor="k",
               label="frontera de Pareto", zorder=3)
    ax.plot(par[ax_x], par[ax_y], "--", c="#d62728", alpha=0.6, zorder=1)
    for _, r in par.iterrows():
        ax.annotate(r["tag"], (r[ax_x], r[ax_y]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel(ax_x + "  (→ mejor)"); ax.set_ylabel(ax_y + "  (↑ mejor)")
    ax.set_title(f"Frontera de Pareto — sweep '{name}'\n(cada punto = una config, en su mejor checkpoint)")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"📊 Pareto guardado en '{out_png}'")


def plot_param_effects(df, keys, ax_x, ax_y, out_png):
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, k in zip(axes[0], keys):
        d = df.copy(); d[k] = d[k].astype(str)
        g = d.groupby(k)[[ax_x, ax_y]].mean().reset_index()
        x = np.arange(len(g))
        ax.plot(x, g[ax_x], "o-", color="#1f77b4", label=ax_x)
        ax2 = ax.twinx()
        ax2.plot(x, g[ax_y], "s--", color="#d62728", label=ax_y)
        ax.set_xticks(x); ax.set_xticklabels(g[k], rotation=20)
        ax.set_xlabel(k); ax.set_ylabel(ax_x, color="#1f77b4"); ax2.set_ylabel(ax_y, color="#d62728")
        ax.set_title(f"efecto de {k}"); ax.grid(True, alpha=0.3)
    fig.suptitle("Efecto marginal de cada parámetro (promediando el resto)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"📊 Efectos por parámetro en '{out_png}'")


# =====================================================================
# DEFINICIÓN DEL SWEEP (editá acá)
# =====================================================================
def model09_sweep():
    base = Config(
        train_csv="dataset_train_signed.csv", val_csv="dataset_val_signed.csv",
        neck_hidden=[256], dropout=0.4, backbone_frozen=True,
        batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=15,
        scheduler_step=7, scheduler_gamma=0.1,
        use_class_weights=True, class_weight_power=0.5, lambda_stop=1.0,
    )
    grid = {                                   # Tier 1 (knobs de la falla) + Tier 2 (capacidad)
        "lambda_stop": [0.3, 0.5, 1.0],        # peso de la BCE is-stop (barre el Pareto)
        "stop_pos_weight_cap": [2.0, 3.5, 6.5],  # tope al pos_weight del motor B (sobre-stop)
        "neck_hidden": [[256], [512, 256]],    # capacidad del cuello
        "dropout": [0.4, 0.5],                 # regularización
    }                                          # 3*3*2*2 = 36 configs
    return dict(name="m09_tier2", engine="hybrid", base=base, grid=grid,
                axis_x="val_dir_f1_macro", axis_y="val_fullstop_recall",
                reseed_seeds=(101, 202))


def model01_sweep():
    """Sweep del per-frame (model_01, el mejor): arquitectura + regularización + optimización."""
    base = Config(                                 # = model_01_perframe (bases dir+speed por defecto)
        neck_hidden=[256], dropout=0.4, backbone_frozen=True,
        batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=15,
        scheduler_step=7, scheduler_gamma=0.1,
        use_class_weights=True, lambda_speed=1.0,
    )
    grid = {
        "neck_hidden": [[256], [512, 256]],        # capacidad del cuello
        "dropout": [0.3, 0.4, 0.5],                # regularización
        "lr": [5e-4, 1e-3],                        # optimización
        "weight_decay": [1e-4, 1e-3],              # regularización L2
    }                                              # 2*3*2*2 = 24 configs
    return dict(name="m01_perframe", engine="perframe", base=base, grid=grid,
                axis_x="val_dir_f1_macro", axis_y="val_fullstop_recall",
                reseed_seeds=(101, 202))


SWEEPS = {"m09": model09_sweep, "m01": model01_sweep}


def smoke_sweep():
    """Mini-sweep para validar el pipeline end-to-end (2 configs, 2 épocas, 1 reseed)."""
    spec = model09_sweep()
    spec["base"] = replace(spec["base"], max_epochs=2)
    spec["grid"] = {"lambda_stop": [0.5, 1.0]}
    spec["name"] = "smoke"
    spec["reseed_seeds"] = (101,)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--which", default="m09", choices=list(SWEEPS))  # qué sweep correr
    ap.add_argument("--engine"); ap.add_argument("--tag"); ap.add_argument("--results-dir")
    ap.add_argument("--seed"); ap.add_argument("--cfg")
    args = ap.parse_args()
    if args.run_one:
        run_one(args.engine, args.tag, args.results_dir, args.seed, args.cfg)
    elif args.smoke:
        run_sweep(**smoke_sweep())
    else:
        run_sweep(**SWEEPS[args.which]())


if __name__ == "__main__":
    main()

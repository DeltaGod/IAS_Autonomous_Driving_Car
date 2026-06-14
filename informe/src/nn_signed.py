"""
nn_signed.py -- MOTOR de entrenamiento PER-FRAME con VELOCIDAD CON SIGNO (no es entrypoint).
============================================================================================
Modelo: IMAGEN -> 2 variables CONTINUAS (signed_speedA, signed_speedB)  in  [-100, 100]:
    > 0 FORWARD   < 0 BACKWARD   = 0 STOP   (la direccion es el SIGNO de la salida)

Diferencia con nn_perframe: NO hay cabezas de clasificacion de direccion. Una sola salida
continua por motor (tanh*100) unifica direccion + velocidad en el comando fisico (PWM).

PARA QUE BACKWARD SOBREVIVA (sin class-weights, porque no hay clases): la L1 se PONDERA
POR MUESTRA segun la direccion REAL de cada motor (peso inverso a la frecuencia, igual
idea que las class-weights del modelo por-frame, pero aplicado al termino L1 de cada motor).

METRICAS (val, calculadas sobre la salida continua):
  - val_signed_mae_A/B/avg : MAE del PWM con signo (la metrica nativa; titular).
  - val_dir_f1_A/B/macro   : F1 macro decodificando sign(salida) -> {STOP,FWD,BWD},
                             para poder COMPARAR con los modelos 00-05 del scoreboard.
  - val_fullstop_recall    : recall del escenario "ambos motores STOP" (seguridad).

Este archivo NO se corre directo: expone `run_experiment(cfg, tag)`. El modelo concreto
vive en model_06_signed.py con su Config congelada (lee dataset_*_signed.csv).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from PIL import Image

from config import active_params_text
from nn_perframe import (
    DIR_MAP, DIR_INV, SPEED_SCALE, TURNS, compute_class_weights,
    save_config_snapshot, make_trainer,
)

DIR_EPS_PWM = 1.0  # |signed| < 1 PWM -> STOP (mismo umbral que el harness de inferencia)


def signed_to_dir(v_norm, eps_norm):
    """Decodifica un PWM-con-signo NORMALIZADO (/100) a direccion {0:STOP,1:FWD,2:BWD}."""
    out = np.zeros_like(v_norm, dtype=int)          # STOP
    out[v_norm > eps_norm] = DIR_MAP["FORWARD"]     # 1
    out[v_norm < -eps_norm] = DIR_MAP["BACKWARD"]   # 2
    return out


# ==========================================
# 1. DATASET (target = signed speed; flip = swap A<->B con signo intacto)
# ==========================================
class SignedDriveDataset(Dataset):
    def __init__(self, df, is_train=True, cfg=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.cfg = cfg
        self.resize = T.Resize((cfg.image_size, cfg.image_size), antialias=True)
        self.jitter = T.ColorJitter(cfg.jitter_brightness, cfg.jitter_contrast,
                                    cfg.jitter_saturation, cfg.jitter_hue)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.index = self._build_resampled_index() if is_train else np.arange(len(self.df))

    def _build_resampled_index(self):
        rng = np.random.default_rng(self.cfg.seed)
        factors = self.cfg.resample_factor
        beh = self.df["behavior"].values
        idxs = []
        for i in range(len(self.df)):
            f = float(factors.get(beh[i], 1.0))
            n = int(f)
            idxs.extend([i] * n)
            frac = f - n
            if frac > 0 and rng.random() < frac:
                idxs.append(i)
        return np.array(idxs, dtype=int)

    def resampled_frames(self):
        return self.df.iloc[self.index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        row = self.df.iloc[self.index[k]]
        img = Image.open(row['image_path']).convert('RGB')
        sA, sB = float(row['signed_speedA']), float(row['signed_speedB'])
        dirA, dirB = row['behaviorA'], row['behaviorB']

        img = self.resize(img)
        if self.is_train:
            img = self.jitter(img)
            p = self.cfg.flip_prob_turns if row['behavior'] in TURNS else self.cfg.flip_prob_straight
            if torch.rand(1).item() < p:
                img = TF.hflip(img)
                sA, sB = sB, sA       # swap motor A<->B; el SIGNO no cambia (fwd/bwd es el eje)
                dirA, dirB = dirB, dirA

        img = self.normalize(TF.to_tensor(img))
        return img, {
            "spdA": torch.tensor(sA / SPEED_SCALE, dtype=torch.float32),
            "spdB": torch.tensor(sB / SPEED_SCALE, dtype=torch.float32),
            "dirA": torch.tensor(DIR_MAP[dirA], dtype=torch.long),  # solo para peso + metrica
            "dirB": torch.tensor(DIR_MAP[dirB], dtype=torch.long),
        }


# ==========================================
# 2. MODELO (MobileNet congelado -> cuello -> 2 salidas tanh)
# ==========================================
class SignedControlNet(pl.LightningModule):
    def __init__(self, neck_hidden=(256,), dropout=0.4, backbone_frozen=True,
                 lr=1e-3, weight_decay=1e-4, scheduler_step=7, scheduler_gamma=0.1,
                 dir_weights=None, eps_norm=DIR_EPS_PWM / SPEED_SCALE):
        super().__init__()
        self.save_hyperparameters(ignore=['dir_weights'])

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        feat_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        if backbone_frozen:
            for p in backbone.features.parameters():
                p.requires_grad = False
        self.backbone = backbone

        layers, d = [], feat_dim
        for h in neck_hidden:
            layers += [nn.Linear(d, h), nn.Hardswish(), nn.Dropout(dropout)]
            d = h
        self.neck = nn.Sequential(*layers)
        self.head_spdA = nn.Linear(d, 1)
        self.head_spdB = nn.Linear(d, 1)

        # Pesos por direccion (STOP/FWD/BWD) para ponderar la L1 por muestra.
        if dir_weights is None:
            dir_weights = torch.ones(3)
        self.register_buffer("dir_weights", dir_weights)
        self._reset_val_store()

    def _reset_val_store(self):
        self.val_store = {k: [] for k in ["spdA_p", "spdA_t", "spdB_p", "spdB_t",
                                          "dirA_t", "dirB_t"]}

    def forward(self, x):
        z = self.neck(self.backbone(x))
        # tanh acota la salida a [-1,1] -> *no* multiplicamos aca; el target esta /100 en [-1,1].
        return torch.tanh(self.head_spdA(z)).squeeze(1), torch.tanh(self.head_spdB(z)).squeeze(1)

    def _weighted_l1(self, out, tgt):
        pA, pB = out
        wA = self.dir_weights[tgt["dirA"]]   # peso por la direccion REAL de cada motor
        wB = self.dir_weights[tgt["dirB"]]
        loss = (wA * (pA - tgt["spdA"]).abs()).mean() + (wB * (pB - tgt["spdB"]).abs()).mean()
        return loss

    def training_step(self, batch, batch_idx):
        x, tgt = batch
        loss = self._weighted_l1(self(x), tgt)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._reset_val_store()

    def validation_step(self, batch, batch_idx):
        x, tgt = batch
        pA, pB = self(x)
        self.log('val_loss', self._weighted_l1((pA, pB), tgt), on_step=False, on_epoch=True, prog_bar=True)
        s = self.val_store
        s["spdA_p"].append(pA.float().cpu()); s["spdA_t"].append(tgt["spdA"].cpu())
        s["spdB_p"].append(pB.float().cpu()); s["spdB_t"].append(tgt["spdB"].cpu())
        s["dirA_t"].append(tgt["dirA"].cpu()); s["dirB_t"].append(tgt["dirB"].cpu())

    def on_validation_epoch_end(self):
        s = {k: torch.cat(v).numpy() for k, v in self.val_store.items()}
        eps = self.hparams.eps_norm
        maeA = np.abs(s["spdA_p"] - s["spdA_t"]).mean() * SPEED_SCALE
        maeB = np.abs(s["spdB_p"] - s["spdB_t"]).mean() * SPEED_SCALE
        # decodificar direccion desde el signo para F1 comparable con los modelos 00-05
        pdA, pdB = signed_to_dir(s["spdA_p"], eps), signed_to_dir(s["spdB_p"], eps)
        tdA, tdB = s["dirA_t"], s["dirB_t"]
        f1A = f1_score(tdA, pdA, average='macro', labels=[0, 1, 2], zero_division=0)
        f1B = f1_score(tdB, pdB, average='macro', labels=[0, 1, 2], zero_division=0)
        # recall de full-stop conjunto (ambos motores STOP)
        true_stop = (tdA == 0) & (tdB == 0)
        pred_stop = (pdA == 0) & (pdB == 0)
        fs_recall = float((pred_stop & true_stop).sum() / true_stop.sum()) if true_stop.sum() else 0.0

        self.log('val_signed_mae_A', maeA, prog_bar=True)
        self.log('val_signed_mae_B', maeB, prog_bar=True)
        self.log('val_signed_mae_avg', (maeA + maeB) / 2, prog_bar=True)
        self.log('val_dir_f1_A', f1A)
        self.log('val_dir_f1_B', f1B)
        self.log('val_dir_f1_macro', (f1A + f1B) / 2, prog_bar=True)
        self.log('val_fullstop_recall', fs_recall, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.hparams.scheduler_step,
                                                gamma=self.hparams.scheduler_gamma)
        return [opt], [sched]


# ==========================================
# 3. GRAFICADO + EVALUACION
# ==========================================
def plot_training_progress(metrics_csv, out_png, cfg=None):
    if not os.path.exists(metrics_csv):
        print(f"[WARN] No se encontro {metrics_csv}.")
        return
    m = pd.read_csv(metrics_csv).groupby("epoch").mean(numeric_only=True).reset_index()
    panels = [
        ("Loss (L1 ponderada)", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("MAE PWM con signo (Val)", [("val_signed_mae_A", "Motor A"), ("val_signed_mae_B", "Motor B")]),
        ("F1 macro direccion decodificada (Val)", [("val_dir_f1_A", "Motor A"), ("val_dir_f1_B", "Motor B")]),
        ("Recall full-stop (Val)", [("val_fullstop_recall", "full-stop")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Progreso -- SignedControlNet (PWM con signo)", fontsize=15)
    for ax, (title, series) in zip(axes.ravel(), panels):
        plotted = False
        for col, label in series:
            if col in m.columns and m[col].notna().any():
                d = m[["epoch", col]].dropna()
                ax.plot(d["epoch"], d[col], marker="o", label=label); plotted = True
        ax.set_title(title); ax.set_xlabel("Epoca"); ax.grid(True, alpha=0.3)
        ax.legend() if plotted else ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes)
    fig.text(0.5, 0.005, "Parametros: " + active_params_text(cfg).replace("\n", "  |  "),
             ha="center", fontsize=8, family="monospace")
    plt.tight_layout(rect=[0, 0.03, 1, 1]); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"\n[graf] Curvas guardadas en '{out_png}'")


@torch.no_grad()
def evaluate_and_plot(model, loader, device, out_png, report_path=None, cfg=None):
    model.eval().to(device)
    eps = model.hparams.eps_norm
    pA, tA, pB, tB, eA, eB = [], [], [], [], [], []
    tdA, tdB = [], []
    for x, tgt in loader:
        a, b = model(x.to(device))
        pA.append(a.cpu()); tA.append(tgt["spdA"]); pB.append(b.cpu()); tB.append(tgt["spdB"])
        eA.append((a.cpu() - tgt["spdA"]).abs()); eB.append((b.cpu() - tgt["spdB"]).abs())
        tdA.append(tgt["dirA"]); tdB.append(tgt["dirB"])
    pA, tB_ = torch.cat(pA).numpy(), torch.cat(tB).numpy()
    pB = torch.cat(pB).numpy()
    tdA, tdB = torch.cat(tdA).numpy(), torch.cat(tdB).numpy()
    pdA, pdB = signed_to_dir(pA, eps), signed_to_dir(pB, eps)
    maeA = round(float(torch.cat(eA).mean()) * SPEED_SCALE, 2)
    maeB = round(float(torch.cat(eB).mean()) * SPEED_SCALE, 2)
    names = [DIR_INV[i] for i in range(3)]

    true_stop = (tdA == 0) & (tdB == 0)
    pred_stop = (pdA == 0) & (pdB == 0)
    fs = float((pred_stop & true_stop).sum() / true_stop.sum()) if true_stop.sum() else 0.0

    lines = []
    lines.append("===== MOTOR A (IZQ) -- direccion decodificada de sign(PWM) =====")
    lines.append(classification_report(tdA, pdA, labels=[0, 1, 2], target_names=names, zero_division=0))
    lines.append(f"MAE PWM con signo A: {maeA}")
    lines.append("\n===== MOTOR B (DER) -- direccion decodificada de sign(PWM) =====")
    lines.append(classification_report(tdB, pdB, labels=[0, 1, 2], target_names=names, zero_division=0))
    lines.append(f"MAE PWM con signo B: {maeB}")
    lines.append(f"\nRecall full-stop conjunto (ambos motores STOP): {fs:.3f}")
    lines.append(f"MAE PWM con signo PROMEDIO (titular): {round((maeA + maeB) / 2, 2)}")
    report = "\n".join(lines)
    print("\n" + report)
    if report_path:
        with open(report_path, "w") as f:
            f.write(report + "\n")
        print(f"[txt] Reporte guardado en '{report_path}'")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (p, t, title) in zip(axes, [(pdA, tdA, "Motor A (IZQ)"), (pdB, tdB, "Motor B (DER)")]):
        sns.heatmap(confusion_matrix(t, p, labels=[0, 1, 2]), annot=True, fmt='d', cmap='Blues',
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(f"Confusion (dir decodificada) -- {title}"); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    fig.suptitle("model_06_signed -- direccion = signo del PWM predicho")
    plt.tight_layout(rect=[0, 0.04, 1, 1]); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[graf] Matrices de confusion guardadas en '{out_png}'")


# ==========================================
# 4. PIPELINE
# ==========================================
def run_experiment(cfg, tag, results_dir="results", select_by="mae"):
    """Entrena el modelo SIGNED con la `cfg` congelada y deja TODO en results/<tag>/.

    select_by: criterio para elegir el MEJOR checkpoint:
      "mae" -> menor val_signed_mae_avg (metrica fisica nativa).
      "f1"  -> mayor val_dir_f1_macro (direccion decodificada; comparable con 00-05).
    """
    torch.set_float32_matmul_precision('medium')
    out = os.path.join(results_dir, tag)
    os.makedirs(out, exist_ok=True)
    save_config_snapshot(cfg, os.path.join(out, "config_used.txt"),
                         extra=f"# {tag} -- PER-FRAME, velocidad CON SIGNO (nn_signed.py)")

    print(f"=== {tag} | SIGNED === Cargando CSVs...")
    train_df = pd.read_csv(cfg.train_csv)
    val_df = pd.read_csv(cfg.val_csv)
    train_ds = SignedDriveDataset(train_df, is_train=True, cfg=cfg)
    val_ds = SignedDriveDataset(val_df, is_train=False, cfg=cfg)
    print(f"Frames -> Train: {len(train_df)} (efectivos tras resample: {len(train_ds)}) | Val: {len(val_df)}")
    print("Parametros activos:\n" + active_params_text(cfg))

    # Pesos por direccion para la L1 ponderada (que BACKWARD sobreviva), SUAVIZADOS
    # por class_weight_power (1=inverso a freq, 0.5=raiz, 0=uniforme) para no aplastar STOP.
    dir_weights = compute_class_weights(train_ds.resampled_frames()) if cfg.use_class_weights else torch.ones(3)
    power = getattr(cfg, "class_weight_power", 1.0)
    if power != 1.0:
        dir_weights = dir_weights ** power
    print(f"Pesos L1 por direccion (power={power}) {[DIR_INV[i] for i in range(3)]}: {dir_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = SignedControlNet(
        neck_hidden=tuple(cfg.neck_hidden), dropout=cfg.dropout, backbone_frozen=cfg.backbone_frozen,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        scheduler_step=cfg.scheduler_step, scheduler_gamma=cfg.scheduler_gamma,
        dir_weights=dir_weights,
    )

    # Seleccion del mejor checkpoint segun select_by. make_trainer ya viene con F1 (max);
    # solo lo cambiamos a MAE si se pide.
    trainer, csv_logger, checkpoint_callback = make_trainer(cfg, out, ckpt_prefix="motorsigned")
    if select_by == "mae":
        checkpoint_callback.monitor = "val_signed_mae_avg"
        checkpoint_callback.mode = "min"
        checkpoint_callback.filename = "motorsigned-{epoch:02d}-{val_signed_mae_avg:.2f}"
    else:  # "f1" -> usa el default de make_trainer (val_dir_f1_macro, max)
        checkpoint_callback.filename = "motorsigned-{epoch:02d}-{val_dir_f1_macro:.3f}"
    print(f"Seleccion de mejor epoca por: {select_by} "
          f"({checkpoint_callback.monitor}, mode={checkpoint_callback.mode})")

    print(f"\n[run] ENTRENANDO {tag} (SignedControlNet) [run]")
    trainer.fit(model, train_loader, val_loader)

    plot_training_progress(os.path.join(csv_logger.log_dir, "metrics.csv"),
                           out_png=os.path.join(out, "training_progress.png"), cfg=cfg)
    best = checkpoint_callback.best_model_path
    if best and os.path.exists(best):
        print(f"\nEvaluando mejor checkpoint: {best}")
        model = SignedControlNet.load_from_checkpoint(best, dir_weights=dir_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_and_plot(model, val_loader, device,
                      out_png=os.path.join(out, "eval_confusion.png"),
                      report_path=os.path.join(out, "report.txt"), cfg=cfg)
    print(f"\n[OK] {tag} listo. Resultados en {out}/")

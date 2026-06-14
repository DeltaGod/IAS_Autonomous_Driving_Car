"""
nn_hybrid.py -- MOTOR per-frame HIBRIDO: signed regression + cabeza "is-stop" (no es entrypoint).
==================================================================================================
Arregla el colapso de STOP del modelo signed (model_06/08): la regresion-a-cero no se
detecta por umbral, asi que le agregamos una cabeza de CLASIFICACION binaria explicita
"este motor esta parado?" entrenada con BCE -- que SI tiene incentivo para detectar el cero.

Por motor, 2 salidas:
  - signed (tanh, PWM con signo)         -> direccion+magnitud CUANDO se mueve
  - is_stop (logit, binaria)             -> esta parado? (la autoridad sobre STOP)

DECODE:  STOP si sigmoid(is_stop) > 0.5 ; si no, dir = signo(signed), speed = |signed|.

Loss = L1_ponderada(signed) + lambda_stop * BCE(is_stop, pos_weight por motor).
La BCE usa pos_weight = #no_stop/#stop por motor (motor B casi no para -> pesa mas).

Metricas (val): F1 macro de direccion decodificada (comparable con 00-08), MAE-PWM, full-stop recall.
Seleccion de mejor epoca por F1 macro. Reusa el Dataset signed de nn_signed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from config import active_params_text
from nn_perframe import (
    DIR_MAP, DIR_INV, SPEED_SCALE, compute_class_weights, save_config_snapshot, make_trainer,
)
from nn_signed import SignedDriveDataset, signed_to_dir, DIR_EPS_PWM


def decode_hybrid(signed_norm, stop_prob, eps_norm):
    """STOP si stop_prob>0.5; si no, direccion = signo del PWM. -> {0:STOP,1:FWD,2:BWD}."""
    out = signed_to_dir(signed_norm, eps_norm)   # 1/2 segun signo (0 si |v|<eps)
    out[stop_prob > 0.5] = 0                      # la cabeza is-stop manda
    return out


# ==========================================
# MODELO: backbone congelado -> cuello -> 4 cabezas (2 signed + 2 is-stop)
# ==========================================
class HybridControlNet(pl.LightningModule):
    def __init__(self, neck_hidden=(256,), dropout=0.4, backbone_frozen=True,
                 lr=1e-3, weight_decay=1e-4, scheduler_step=7, scheduler_gamma=0.1,
                 lambda_stop=1.0, dir_weights=None, stop_pos_weight=None,
                 eps_norm=DIR_EPS_PWM / SPEED_SCALE):
        super().__init__()
        self.save_hyperparameters(ignore=['dir_weights', 'stop_pos_weight'])

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
        self.head_stopA = nn.Linear(d, 1)
        self.head_stopB = nn.Linear(d, 1)

        if dir_weights is None:
            dir_weights = torch.ones(3)
        if stop_pos_weight is None:
            stop_pos_weight = torch.ones(2)
        self.register_buffer("dir_weights", dir_weights)
        self.register_buffer("stop_pos_weight", stop_pos_weight)
        self._reset_val_store()

    def _reset_val_store(self):
        self.val_store = {k: [] for k in ["sA", "sB", "stA", "stB", "dirA_t", "dirB_t",
                                          "spdA_t", "spdB_t"]}

    def forward(self, x):
        z = self.neck(self.backbone(x))
        return (torch.tanh(self.head_spdA(z)).squeeze(1),
                torch.tanh(self.head_spdB(z)).squeeze(1),
                self.head_stopA(z).squeeze(1),
                self.head_stopB(z).squeeze(1))

    def _loss(self, out, tgt):
        sA, sB, stA, stB = out
        is_stopA = (tgt["dirA"] == 0).float()
        is_stopB = (tgt["dirB"] == 0).float()
        wA, wB = self.dir_weights[tgt["dirA"]], self.dir_weights[tgt["dirB"]]
        l1 = (wA * (sA - tgt["spdA"]).abs()).mean() + (wB * (sB - tgt["spdB"]).abs()).mean()
        bce = (F.binary_cross_entropy_with_logits(stA, is_stopA, pos_weight=self.stop_pos_weight[0])
               + F.binary_cross_entropy_with_logits(stB, is_stopB, pos_weight=self.stop_pos_weight[1]))
        return l1 + self.hparams.lambda_stop * bce, l1, bce

    def training_step(self, batch, batch_idx):
        x, tgt = batch
        loss, l1, bce = self._loss(self(x), tgt)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_l1', l1, on_step=False, on_epoch=True)
        self.log('train_bce', bce, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._reset_val_store()

    def validation_step(self, batch, batch_idx):
        x, tgt = batch
        out = self(x)
        loss, _, _ = self._loss(out, tgt)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        sA, sB, stA, stB = out
        s = self.val_store
        s["sA"].append(sA.float().cpu()); s["sB"].append(sB.float().cpu())
        s["stA"].append(torch.sigmoid(stA).cpu()); s["stB"].append(torch.sigmoid(stB).cpu())
        s["dirA_t"].append(tgt["dirA"].cpu()); s["dirB_t"].append(tgt["dirB"].cpu())
        s["spdA_t"].append(tgt["spdA"].cpu()); s["spdB_t"].append(tgt["spdB"].cpu())

    def on_validation_epoch_end(self):
        s = {k: torch.cat(v).numpy() for k, v in self.val_store.items()}
        eps = self.hparams.eps_norm
        pdA = decode_hybrid(s["sA"], s["stA"], eps)
        pdB = decode_hybrid(s["sB"], s["stB"], eps)
        tdA, tdB = s["dirA_t"], s["dirB_t"]
        f1A = f1_score(tdA, pdA, average='macro', labels=[0, 1, 2], zero_division=0)
        f1B = f1_score(tdB, pdB, average='macro', labels=[0, 1, 2], zero_division=0)
        maeA = np.abs(s["sA"] - s["spdA_t"]).mean() * SPEED_SCALE
        maeB = np.abs(s["sB"] - s["spdB_t"]).mean() * SPEED_SCALE
        true_stop = (tdA == 0) & (tdB == 0)
        pred_stop = (pdA == 0) & (pdB == 0)
        fs = float((pred_stop & true_stop).sum() / true_stop.sum()) if true_stop.sum() else 0.0
        self.log('val_signed_mae_A', maeA); self.log('val_signed_mae_B', maeB)
        self.log('val_signed_mae_avg', (maeA + maeB) / 2, prog_bar=True)
        self.log('val_f1_A', f1A); self.log('val_f1_B', f1B)
        self.log('val_dir_f1_macro', (f1A + f1B) / 2, prog_bar=True)
        self.log('val_fullstop_recall', fs, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.hparams.scheduler_step,
                                                gamma=self.hparams.scheduler_gamma)
        return [opt], [sched]


# ==========================================
# GRAFICADO + EVALUACION
# ==========================================
def plot_training_progress(metrics_csv, out_png, cfg=None):
    if not os.path.exists(metrics_csv):
        print(f"[WARN] No se encontro {metrics_csv}."); return
    m = pd.read_csv(metrics_csv).groupby("epoch").mean(numeric_only=True).reset_index()
    panels = [
        ("Loss total", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("F1 macro direccion (Val)", [("val_f1_A", "Motor A"), ("val_f1_B", "Motor B")]),
        ("MAE PWM con signo (Val)", [("val_signed_mae_A", "Motor A"), ("val_signed_mae_B", "Motor B")]),
        ("Recall full-stop (Val)", [("val_fullstop_recall", "full-stop")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Progreso -- HybridControlNet (signed + is-stop)", fontsize=15)
    for ax, (title, series) in zip(axes.ravel(), panels):
        plotted = False
        for col, label in series:
            if col in m.columns and m[col].notna().any():
                d = m[["epoch", col]].dropna(); ax.plot(d["epoch"], d[col], marker="o", label=label); plotted = True
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
    sA_, sB_, stA_, stB_, tdA_, tdB_, eA, eB = [], [], [], [], [], [], [], []
    for x, tgt in loader:
        a, b, sa, sb = model(x.to(device))
        sA_.append(a.cpu()); sB_.append(b.cpu())
        stA_.append(torch.sigmoid(sa).cpu()); stB_.append(torch.sigmoid(sb).cpu())
        tdA_.append(tgt["dirA"]); tdB_.append(tgt["dirB"])
        eA.append((a.cpu() - tgt["spdA"]).abs()); eB.append((b.cpu() - tgt["spdB"]).abs())
    sA = torch.cat(sA_).numpy(); sB = torch.cat(sB_).numpy()
    stA = torch.cat(stA_).numpy(); stB = torch.cat(stB_).numpy()
    tdA = torch.cat(tdA_).numpy(); tdB = torch.cat(tdB_).numpy()
    pdA = decode_hybrid(sA, stA, eps); pdB = decode_hybrid(sB, stB, eps)
    maeA = round(float(torch.cat(eA).mean()) * SPEED_SCALE, 2)
    maeB = round(float(torch.cat(eB).mean()) * SPEED_SCALE, 2)
    names = [DIR_INV[i] for i in range(3)]
    true_stop = (tdA == 0) & (tdB == 0); pred_stop = (pdA == 0) & (pdB == 0)
    fs = float((pred_stop & true_stop).sum() / true_stop.sum()) if true_stop.sum() else 0.0

    lines = ["===== MOTOR A (IZQ) -- direccion (is-stop + signo) ====="]
    lines.append(classification_report(tdA, pdA, labels=[0, 1, 2], target_names=names, zero_division=0))
    lines.append(f"MAE PWM con signo A: {maeA}")
    lines.append("\n===== MOTOR B (DER) -- direccion (is-stop + signo) =====")
    lines.append(classification_report(tdB, pdB, labels=[0, 1, 2], target_names=names, zero_division=0))
    lines.append(f"MAE PWM con signo B: {maeB}")
    lines.append(f"\nRecall full-stop conjunto (ambos motores STOP): {fs:.3f}")
    lines.append(f"MAE PWM con signo PROMEDIO: {round((maeA + maeB) / 2, 2)}")
    report = "\n".join(lines)
    print("\n" + report)
    if report_path:
        open(report_path, "w").write(report + "\n"); print(f"[txt] Reporte guardado en '{report_path}'")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (p, t, title) in zip(axes, [(pdA, tdA, "Motor A (IZQ)"), (pdB, tdB, "Motor B (DER)")]):
        sns.heatmap(confusion_matrix(t, p, labels=[0, 1, 2]), annot=True, fmt='d', cmap='Blues',
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(f"Confusion -- {title}"); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    fig.suptitle("model_09_hybrid -- direccion = is-stop + signo del PWM")
    plt.tight_layout(rect=[0, 0.04, 1, 1]); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[graf] Matrices de confusion guardadas en '{out_png}'")


# ==========================================
# PIPELINE
# ==========================================
def run_experiment(cfg, tag, results_dir="results"):
    torch.set_float32_matmul_precision('medium')
    out = os.path.join(results_dir, tag)
    os.makedirs(out, exist_ok=True)
    save_config_snapshot(cfg, os.path.join(out, "config_used.txt"),
                         extra=f"# {tag} -- PER-FRAME HIBRIDO signed + is-stop (nn_hybrid.py)")

    print(f"=== {tag} | HIBRIDO === Cargando CSVs...")
    train_df = pd.read_csv(cfg.train_csv); val_df = pd.read_csv(cfg.val_csv)
    train_ds = SignedDriveDataset(train_df, is_train=True, cfg=cfg)
    val_ds = SignedDriveDataset(val_df, is_train=False, cfg=cfg)
    print(f"Frames -> Train: {len(train_df)} (efectivos: {len(train_ds)}) | Val: {len(val_df)}")
    print("Parametros activos:\n" + active_params_text(cfg))

    # pesos L1 por direccion (suavizados) + pos_weight de la BCE is-stop por motor.
    frames = train_ds.resampled_frames()
    dir_weights = compute_class_weights(frames) if cfg.use_class_weights else torch.ones(3)
    power = getattr(cfg, "class_weight_power", 1.0)
    if power != 1.0:
        dir_weights = dir_weights ** power
    stop_pw = []
    for col in ["behaviorA", "behaviorB"]:
        p_stop = (frames[col] == "STOP").mean()
        stop_pw.append((1 - p_stop) / p_stop if p_stop > 0 else 1.0)
    cap = getattr(cfg, "stop_pos_weight_cap", 0.0)
    if cap and cap > 0:
        stop_pw = [min(x, cap) for x in stop_pw]   # tope para que el motor B no sobre-pare
    stop_pos_weight = torch.tensor(stop_pw, dtype=torch.float32)
    print(f"Pesos L1 dir (power={power}): {[round(x,2) for x in dir_weights.tolist()]} | "
          f"pos_weight is-stop [A,B]: {[round(x,2) for x in stop_pw]}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = HybridControlNet(
        neck_hidden=tuple(cfg.neck_hidden), dropout=cfg.dropout, backbone_frozen=cfg.backbone_frozen,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        scheduler_step=cfg.scheduler_step, scheduler_gamma=cfg.scheduler_gamma,
        lambda_stop=cfg.lambda_stop, dir_weights=dir_weights, stop_pos_weight=stop_pos_weight,
    )

    # mejor epoca por F1 macro (default de make_trainer).
    trainer, csv_logger, checkpoint_callback = make_trainer(cfg, out, ckpt_prefix="motorhybrid")
    checkpoint_callback.filename = "motorhybrid-{epoch:02d}-{val_dir_f1_macro:.3f}"
    print("Seleccion de mejor epoca por: f1 (val_dir_f1_macro, max)")

    print(f"\n[run] ENTRENANDO {tag} (HybridControlNet) [run]")
    trainer.fit(model, train_loader, val_loader)

    plot_training_progress(os.path.join(csv_logger.log_dir, "metrics.csv"),
                           out_png=os.path.join(out, "training_progress.png"), cfg=cfg)
    best = checkpoint_callback.best_model_path
    if best and os.path.exists(best):
        print(f"\nEvaluando mejor checkpoint: {best}")
        model = HybridControlNet.load_from_checkpoint(best, dir_weights=dir_weights,
                                                      stop_pos_weight=stop_pos_weight)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_and_plot(model, val_loader, device,
                      out_png=os.path.join(out, "eval_confusion.png"),
                      report_path=os.path.join(out, "report.txt"), cfg=cfg)
    print(f"\n[OK] {tag} listo. Resultados en {out}/")

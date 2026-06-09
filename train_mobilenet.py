"""
train_mobilenet.py
==================
Modelo: IMAGEN -> control por motor (4 salidas):
  - behaviorA, behaviorB ∈ {STOP, FORWARD, BACKWARD}  (clasificación, 3 clases)
  - speedA, speedB ∈ [0,100]                           (regresión, normalizada a [0,1])

TODOS los parámetros (augmentation, arquitectura, optimización) se leen de `config.py`
(objeto CFG). Para tunear, editá config.py — NO hardcodear acá.

El modelo NO predice GPIO ni el behavior global de 5 clases; se reconstruyen en el
harness determinista (`predictions_to_control`). El flip de augmentation es en ESPACIO
ABSTRACTO (swap motor A<->B): FORWARD/BACKWARD no cambian, LEFT<->RIGHT se intercambian.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from config import CFG, active_params_text

from config import CFG, active_params_text

DIR_MAP = {"STOP": 0, "FORWARD": 1, "BACKWARD": 2}
DIR_INV = {v: k for k, v in DIR_MAP.items()}
SPEED_SCALE = 100.0
TURNS = {"LEFT", "RIGHT"}

# ==========================================
# 1. DATASET (resampling + flip abstracto, todo desde CFG)
# ==========================================
class AutonomousDriveDataset(Dataset):
    def __init__(self, df, is_train=True, cfg=CFG):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.cfg = cfg

        self.resize = T.Resize((cfg.image_size, cfg.image_size), antialias=True)
        self.jitter = T.ColorJitter(cfg.jitter_brightness, cfg.jitter_contrast,
                                    cfg.jitter_saturation, cfg.jitter_hue)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # En train aplicamos los resample_factor por clase (sobre/sub-muestreo); en val, no.
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
        """DataFrame de los frames efectivos (post resample) — para calcular class_weights."""
        return self.df.iloc[self.index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        row = self.df.iloc[self.index[k]]
        img = Image.open(row['image_path']).convert('RGB')
        speedA, speedB = float(row['speedA']), float(row['speedB'])
        dirA, dirB = row['behaviorA'], row['behaviorB']

        img = self.resize(img)

        if self.is_train:
            img = self.jitter(img)
            p = self.cfg.flip_prob_turns if row['behavior'] in TURNS else self.cfg.flip_prob_straight
            if torch.rand(1).item() < p:
                img = TF.hflip(img)
                speedA, speedB = speedB, speedA   # swap motor A<->B
                dirA, dirB = dirB, dirA           # FWD/BWD no cambian; el swap basta

        img = self.normalize(TF.to_tensor(img))
        return img, {
            "dirA": torch.tensor(DIR_MAP[dirA], dtype=torch.long),
            "dirB": torch.tensor(DIR_MAP[dirB], dtype=torch.long),
            "spdA": torch.tensor(speedA / SPEED_SCALE, dtype=torch.float32),
            "spdB": torch.tensor(speedB / SPEED_SCALE, dtype=torch.float32),
        }

# ==========================================
# 2. MODELO MULTI-CABEZA (arquitectura desde CFG)
# ==========================================
class MotorControlNet(pl.LightningModule):
    def __init__(self, neck_hidden=(256,), dropout=0.4, backbone_frozen=True,
                 lr=1e-3, lambda_speed=1.0, weight_decay=1e-4,
                 scheduler_step=7, scheduler_gamma=0.1, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        feat_dim = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Identity()
        if backbone_frozen:
            for p in backbone.features.parameters():
                p.requires_grad = False
        self.backbone = backbone

        # Cuello compartido: una capa (Linear+Hardswish+Dropout) por entrada de neck_hidden.
        layers, d = [], feat_dim
        for h in neck_hidden:
            layers += [nn.Linear(d, h), nn.Hardswish(), nn.Dropout(dropout)]
            d = h
        self.neck = nn.Sequential(*layers)
        self.head_dirA = nn.Linear(d, 3)
        self.head_dirB = nn.Linear(d, 3)
        self.head_spdA = nn.Linear(d, 1)
        self.head_spdB = nn.Linear(d, 1)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.l1 = nn.L1Loss()
        self._reset_val_store()

    def _reset_val_store(self):
        self.val_store = {k: [] for k in ["dirA_p", "dirA_t", "dirB_p", "dirB_t",
                                          "spdA_p", "spdA_t", "spdB_p", "spdB_t"]}

    def forward(self, x):
        z = self.neck(self.backbone(x))
        return (self.head_dirA(z), self.head_dirB(z),
                torch.sigmoid(self.head_spdA(z)).squeeze(1),
                torch.sigmoid(self.head_spdB(z)).squeeze(1))

    def _compute_loss(self, out, tgt):
        logA, logB, spA, spB = out
        ce = nn.functional.cross_entropy
        w = self.class_weights
        loss_dir = ce(logA, tgt["dirA"], weight=w) + ce(logB, tgt["dirB"], weight=w)
        loss_spd = self.l1(spA, tgt["spdA"]) + self.l1(spB, tgt["spdB"])
        return loss_dir + self.hparams.lambda_speed * loss_spd, loss_dir, loss_spd

    def training_step(self, batch, batch_idx):
        x, tgt = batch
        loss, loss_dir, loss_spd = self._compute_loss(self(x), tgt)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_dir', loss_dir, on_step=False, on_epoch=True)
        self.log('train_loss_spd', loss_spd, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._reset_val_store()

    def validation_step(self, batch, batch_idx):
        x, tgt = batch
        out = self(x)
        loss, _, _ = self._compute_loss(out, tgt)
        logA, logB, spA, spB = out
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        s = self.val_store
        s["dirA_p"].append(logA.argmax(1).cpu()); s["dirA_t"].append(tgt["dirA"].cpu())
        s["dirB_p"].append(logB.argmax(1).cpu()); s["dirB_t"].append(tgt["dirB"].cpu())
        s["spdA_p"].append(spA.float().cpu()); s["spdA_t"].append(tgt["spdA"].cpu())
        s["spdB_p"].append(spB.float().cpu()); s["spdB_t"].append(tgt["spdB"].cpu())

    def on_validation_epoch_end(self):
        s = {k: torch.cat(v).numpy() for k, v in self.val_store.items()}
        f1A = f1_score(s["dirA_t"], s["dirA_p"], average='macro', labels=[0, 1, 2], zero_division=0)
        f1B = f1_score(s["dirB_t"], s["dirB_p"], average='macro', labels=[0, 1, 2], zero_division=0)
        maeA = np.abs(s["spdA_p"] - s["spdA_t"]).mean() * SPEED_SCALE
        maeB = np.abs(s["spdB_p"] - s["spdB_t"]).mean() * SPEED_SCALE
        self.log('val_f1_A', f1A, prog_bar=True)
        self.log('val_f1_B', f1B, prog_bar=True)
        self.log('val_dir_f1_macro', (f1A + f1B) / 2, prog_bar=True)
        self.log('val_mae_A', maeA, prog_bar=True)
        self.log('val_mae_B', maeB, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.hparams.scheduler_step,
                                                gamma=self.hparams.scheduler_gamma)
        return [opt], [sched]

# ==========================================
# 3. HARNESS DE INFERENCIA (decoder a pines)
# ==========================================
def predictions_to_control(dirA, dirB, speedA, speedB, speed_eps=1.0):
    """Traduce la salida abstracta a pines del L298N (estados siempre válidos)."""
    if speedA < speed_eps:
        dirA = "STOP"
    if speedB < speed_eps:
        dirB = "STOP"
    gpio = {"FORWARD_A": (1, 0), "BACKWARD_A": (0, 1), "STOP_A": (0, 0),
            "FORWARD_B": (0, 1), "BACKWARD_B": (1, 0), "STOP_B": (0, 0)}
    g1, g2 = gpio[f"{dirA}_A"]; g3, g4 = gpio[f"{dirB}_B"]
    return {"speedA": 0.0 if dirA == "STOP" else round(speedA, 1),
            "speedB": 0.0 if dirB == "STOP" else round(speedB, 1),
            "GPIO1": g1, "GPIO2": g2, "GPIO3": g3, "GPIO4": g4}

# ==========================================
# 4. GRAFICADO LOCAL
# ==========================================
def plot_training_progress(metrics_csv, out_png="training_progress.png"):
    if not os.path.exists(metrics_csv):
        print(f"[WARN] No se encontró {metrics_csv}; no se grafica el progreso.")
        return
    m = pd.read_csv(metrics_csv).groupby("epoch").mean(numeric_only=True).reset_index()
    panels = [
        ("Loss total", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("F1 macro dirección (Val)", [("val_f1_A", "Motor A"), ("val_f1_B", "Motor B")]),
        ("MAE velocidad (Val, 0-100)", [("val_mae_A", "Motor A"), ("val_mae_B", "Motor B")]),
        ("Learning Rate", [("lr-AdamW", "LR")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Progreso de Entrenamiento — MotorControlNet (local)", fontsize=15)
    for ax, (title, series) in zip(axes.ravel(), panels):
        plotted = False
        for col, label in series:
            if col in m.columns and m[col].notna().any():
                d = m[["epoch", col]].dropna()
                ax.plot(d["epoch"], d[col], marker="o", label=label); plotted = True
        ax.set_title(title); ax.set_xlabel("Época"); ax.grid(True, alpha=0.3)
        ax.legend() if plotted else ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes)
    fig.text(0.5, 0.005, "Parámetros: " + active_params_text().replace("\n", "  |  "),
             ha="center", fontsize=8, family="monospace")
    plt.tight_layout(rect=[0, 0.03, 1, 1]); plt.savefig(out_png, dpi=150)
    print(f"\n📈 Curvas guardadas en '{out_png}'")

@torch.no_grad()
def evaluate_and_plot(model, loader, device, out_png="eval_confusion.png"):
    model.eval().to(device)
    pA, tA, pB, tB, eA, eB = [], [], [], [], [], []
    for x, tgt in loader:
        logA, logB, spa, spb = model(x.to(device))
        pA.append(logA.argmax(1).cpu()); tA.append(tgt["dirA"])
        pB.append(logB.argmax(1).cpu()); tB.append(tgt["dirB"])
        eA.append((spa.cpu() - tgt["spdA"]).abs()); eB.append((spb.cpu() - tgt["spdB"]).abs())
    pA, tA = torch.cat(pA).numpy(), torch.cat(tA).numpy()
    pB, tB = torch.cat(pB).numpy(), torch.cat(tB).numpy()
    names = [DIR_INV[i] for i in range(3)]
    print("\n===== MOTOR A (IZQ) =====")
    print(classification_report(tA, pA, labels=[0, 1, 2], target_names=names, zero_division=0))
    print("MAE velocidad A:", round(float(torch.cat(eA).mean()) * SPEED_SCALE, 2))
    print("\n===== MOTOR B (DER) =====")
    print(classification_report(tB, pB, labels=[0, 1, 2], target_names=names, zero_division=0))
    print("MAE velocidad B:", round(float(torch.cat(eB).mean()) * SPEED_SCALE, 2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (p, t, title) in zip(axes, [(pA, tA, "Motor A (IZQ)"), (pB, tB, "Motor B (DER)")]):
        sns.heatmap(confusion_matrix(t, p, labels=[0, 1, 2]), annot=True, fmt='d', cmap='Blues',
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(f"Confusión — {title}"); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    fig.text(0.5, 0.005, "Parámetros: " + active_params_text().replace("\n", "  |  "),
             ha="center", fontsize=8, family="monospace")
    plt.tight_layout(rect=[0, 0.04, 1, 1]); plt.savefig(out_png, dpi=150)
    print(f"\n📊 Matrices de confusión guardadas en '{out_png}'")

# ==========================================
# 5. PIPELINE
# ==========================================
def compute_class_weights(frames):
    pooled = pd.concat([frames["behaviorA"], frames["behaviorB"]]).map(DIR_MAP)
    counts = pooled.value_counts().reindex([0, 1, 2]).fillna(0)
    w = counts.sum() / (3 * counts.replace(0, np.nan))
    return torch.tensor(w.fillna(0).values, dtype=torch.float32)

def main():
    print("Cargando CSVs (rutas desde config.py)...")
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)

    train_ds = AutonomousDriveDataset(train_df, is_train=True, cfg=CFG)
    val_ds = AutonomousDriveDataset(val_df, is_train=False, cfg=CFG)
    print(f"Frames -> Train: {len(train_df)} (efectivos tras resample: {len(train_ds)}) | Val: {len(val_df)}")
    print("Parámetros activos:\n" + active_params_text())

    class_weights = compute_class_weights(train_ds.resampled_frames()) if CFG.use_class_weights else None
    if class_weights is not None:
        print(f"Pesos de clase {[DIR_INV[i] for i in range(3)]}: {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)

    model = MotorControlNet(
        neck_hidden=tuple(CFG.neck_hidden), dropout=CFG.dropout, backbone_frozen=CFG.backbone_frozen,
        lr=CFG.lr, lambda_speed=CFG.lambda_speed, weight_decay=CFG.weight_decay,
        scheduler_step=CFG.scheduler_step, scheduler_gamma=CFG.scheduler_gamma,
        class_weights=class_weights,
    )

    csv_logger = CSVLogger(save_dir="logs", name="MotorControlNet")
    checkpoint_callback = ModelCheckpoint(monitor='val_dir_f1_macro', mode='max', save_top_k=1,
                                          dirpath='checkpoints/',
                                          filename='motorctrl-{epoch:02d}-{val_dir_f1_macro:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(max_epochs=CFG.max_epochs, accelerator='gpu', devices=1, logger=csv_logger,
                         callbacks=[checkpoint_callback, lr_monitor], precision='16-mixed')

    print("\n🚀 ENTRENANDO MotorControlNet 🚀")
    trainer.fit(model, train_loader, val_loader)

    plot_training_progress(os.path.join(csv_logger.log_dir, "metrics.csv"))
    best = checkpoint_callback.best_model_path
    if best and os.path.exists(best):
        print(f"\nEvaluando mejor checkpoint: {best}")
        model = MotorControlNet.load_from_checkpoint(best, class_weights=class_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_and_plot(model, val_loader, device)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()

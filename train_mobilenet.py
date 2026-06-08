"""
train_mobilenet.py
==================
Modelo: IMAGEN -> control por motor.

Salidas del modelo (por motor IZQ=A y DER=B):
  - dirección ∈ {STOP, FORWARD, BACKWARD}  (clasificación, 3 clases)
  - velocidad ∈ [0, 100]                    (regresión, normalizada a [0,1])

El modelo NO predice GPIO ni el behavior global de 5 clases. Esos son derivados:
  - El behavior global (LEFT/RIGHT/...) solo se usa para estadística.
  - Los GPIO se reconstruyen DESPUÉS, en el harness determinista (ver
    predictions_to_control), usando la convención del L298N.

Augmentation flip: se hace en ESPACIO ABSTRACTO (swap motor A<->B de velocidad y
dirección). Espejar la imagen izq/der intercambia los motores; FORWARD/BACKWARD no
cambian (son a lo largo del eje del auto). Esto evita la trampa de swapear bits GPIO.
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

# Dirección por motor -> índice de clase (y su inversa)
DIR_MAP = {"STOP": 0, "FORWARD": 1, "BACKWARD": 2}
DIR_INV = {v: k for k, v in DIR_MAP.items()}
SPEED_SCALE = 100.0  # normalización de velocidad PWM (0-100) a [0,1]

# ==========================================
# 1. DATASET (targets por motor + flip abstracto)
# ==========================================
class AutonomousDriveDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        self.resize = T.Resize((224, 224), antialias=True)
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')

        speedA, speedB = float(row['speedA']), float(row['speedB'])
        dirA, dirB = row['behaviorA'], row['behaviorB']

        img = self.resize(img)

        if self.is_train:
            img = self.jitter(img)
            # Flip horizontal: intercambia motor A<->B (velocidad y dirección).
            # FORWARD/BACKWARD se mantienen; solo cambia QUÉ motor los ejecuta.
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                speedA, speedB = speedB, speedA
                dirA, dirB = dirB, dirA

        img = TF.to_tensor(img)
        img = self.normalize(img)

        target = {
            "dirA": torch.tensor(DIR_MAP[dirA], dtype=torch.long),
            "dirB": torch.tensor(DIR_MAP[dirB], dtype=torch.long),
            "spdA": torch.tensor(speedA / SPEED_SCALE, dtype=torch.float32),
            "spdB": torch.tensor(speedB / SPEED_SCALE, dtype=torch.float32),
        }
        return img, target

# ==========================================
# 2. MODELO MULTI-CABEZA
# ==========================================
class MotorControlNet(pl.LightningModule):
    def __init__(self, lr=1e-3, lambda_speed=1.0, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        feat_dim = backbone.classifier[0].in_features  # 576 en mobilenet_v3_small
        backbone.classifier = nn.Identity()             # nos quedamos con el pooled feature
        # Transfer learning conservador: congelamos todo el extractor de features.
        for p in backbone.features.parameters():
            p.requires_grad = False
        self.backbone = backbone

        # Cuello compartido + 4 cabezas
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(),
            nn.Dropout(0.4),
        )
        self.head_dirA = nn.Linear(256, 3)
        self.head_dirB = nn.Linear(256, 3)
        self.head_spdA = nn.Linear(256, 1)
        self.head_spdB = nn.Linear(256, 1)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.l1 = nn.L1Loss()

        # Buffers para métricas de época (acumulamos y calculamos UNA vez por época)
        self._reset_val_store()

    def _reset_val_store(self):
        self.val_store = {k: [] for k in ["dirA_p", "dirA_t", "dirB_p", "dirB_t",
                                          "spdA_p", "spdA_t", "spdB_p", "spdB_t"]}

    def forward(self, x):
        f = self.backbone(x)
        z = self.neck(f)
        return (
            self.head_dirA(z),
            self.head_dirB(z),
            torch.sigmoid(self.head_spdA(z)).squeeze(1),  # [0,1]
            torch.sigmoid(self.head_spdB(z)).squeeze(1),
        )

    def _compute_loss(self, out, tgt):
        logA, logB, spA, spB = out
        ce = nn.functional.cross_entropy
        w = self.class_weights
        loss_dirA = ce(logA, tgt["dirA"], weight=w)
        loss_dirB = ce(logB, tgt["dirB"], weight=w)
        loss_spd = self.l1(spA, tgt["spdA"]) + self.l1(spB, tgt["spdB"])
        loss = loss_dirA + loss_dirB + self.hparams.lambda_speed * loss_spd
        return loss, loss_dirA + loss_dirB, loss_spd

    def training_step(self, batch, batch_idx):
        x, tgt = batch
        out = self(x)
        loss, loss_dir, loss_spd = self._compute_loss(out, tgt)
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
        # Acumulamos para calcular F1/MAE correctos al final de la época.
        s = self.val_store
        s["dirA_p"].append(logA.argmax(1).cpu()); s["dirA_t"].append(tgt["dirA"].cpu())
        s["dirB_p"].append(logB.argmax(1).cpu()); s["dirB_t"].append(tgt["dirB"].cpu())
        s["spdA_p"].append(spA.float().cpu()); s["spdA_t"].append(tgt["spdA"].cpu())
        s["spdB_p"].append(spB.float().cpu()); s["spdB_t"].append(tgt["spdB"].cpu())

    def on_validation_epoch_end(self):
        s = {k: torch.cat(v).numpy() for k, v in self.val_store.items()}
        # F1 macro por motor (3 clases), bien calculado sobre TODA la época.
        f1A = f1_score(s["dirA_t"], s["dirA_p"], average='macro', labels=[0, 1, 2], zero_division=0)
        f1B = f1_score(s["dirB_t"], s["dirB_p"], average='macro', labels=[0, 1, 2], zero_division=0)
        maeA = np.abs(s["spdA_p"] - s["spdA_t"]).mean() * SPEED_SCALE
        maeB = np.abs(s["spdB_p"] - s["spdB_t"]).mean() * SPEED_SCALE

        self.log('val_f1_A', f1A, prog_bar=True)
        self.log('val_f1_B', f1B, prog_bar=True)
        self.log('val_dir_f1_macro', (f1A + f1B) / 2, prog_bar=True)  # métrica de checkpoint
        self.log('val_mae_A', maeA, prog_bar=True)
        self.log('val_mae_B', maeB, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

# ==========================================
# 3. HARNESS DE INFERENCIA (decoder a pines, para el auto)
# ==========================================
def predictions_to_control(dirA, dirB, speedA, speedB, speed_eps=1.0):
    """Traduce la salida abstracta del modelo a lo que se manda al L298N.
    Garantiza estados válidos (nunca (1,1), nunca speed>0 con freno).
    dirA/dirB ∈ {'STOP','FORWARD','BACKWARD'}; speedA/speedB en 0-100.
    Devuelve dict con speedA, speedB, GPIO1..4."""
    # Consistencia: velocidad ínfima -> tratamos como STOP (regla de freno).
    if speedA < speed_eps:
        dirA = "STOP"
    if speedB < speed_eps:
        dirB = "STOP"

    gpio = {"FORWARD_A": (1, 0), "BACKWARD_A": (0, 1), "STOP_A": (0, 0),
            "FORWARD_B": (0, 1), "BACKWARD_B": (1, 0), "STOP_B": (0, 0)}
    g1, g2 = gpio[f"{dirA}_A"]
    g3, g4 = gpio[f"{dirB}_B"]
    return {
        "speedA": 0.0 if dirA == "STOP" else round(speedA, 1),
        "speedB": 0.0 if dirB == "STOP" else round(speedB, 1),
        "GPIO1": g1, "GPIO2": g2, "GPIO3": g3, "GPIO4": g4,
    }

# ==========================================
# 4. GRAFICADO LOCAL (reemplaza Weights & Biases)
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
                ax.plot(d["epoch"], d[col], marker="o", label=label)
                plotted = True
        ax.set_title(title); ax.set_xlabel("Época"); ax.grid(True, alpha=0.3)
        ax.legend() if plotted else ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout(); plt.savefig(out_png, dpi=150)
    print(f"\n📈 Curvas guardadas en '{out_png}'")

@torch.no_grad()
def evaluate_and_plot(model, loader, device, out_png="eval_confusion.png"):
    """Corre el modelo sobre validación, imprime classification_report por motor y
    grafica las dos matrices de confusión (3x3)."""
    model.eval().to(device)
    pA, tA, pB, tB = [], [], [], []
    spA_e, spB_e = [], []
    for x, tgt in loader:
        logA, logB, spa, spb = model(x.to(device))
        pA.append(logA.argmax(1).cpu()); tA.append(tgt["dirA"])
        pB.append(logB.argmax(1).cpu()); tB.append(tgt["dirB"])
        spA_e.append((spa.cpu() - tgt["spdA"]).abs()); spB_e.append((spb.cpu() - tgt["spdB"]).abs())
    pA, tA = torch.cat(pA).numpy(), torch.cat(tA).numpy()
    pB, tB = torch.cat(pB).numpy(), torch.cat(tB).numpy()
    names = [DIR_INV[i] for i in range(3)]

    print("\n================= MOTOR A (IZQ) =================")
    print(classification_report(tA, pA, labels=[0, 1, 2], target_names=names, zero_division=0))
    print("MAE velocidad A:", round(float(torch.cat(spA_e).mean()) * SPEED_SCALE, 2))
    print("\n================= MOTOR B (DER) =================")
    print(classification_report(tB, pB, labels=[0, 1, 2], target_names=names, zero_division=0))
    print("MAE velocidad B:", round(float(torch.cat(spB_e).mean()) * SPEED_SCALE, 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (p, t, title) in zip(axes, [(pA, tA, "Motor A (IZQ)"), (pB, tB, "Motor B (DER)")]):
        cm = confusion_matrix(t, p, labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(f"Matriz de confusión — {title}")
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    plt.tight_layout(); plt.savefig(out_png, dpi=150)
    print(f"\n📊 Matrices de confusión guardadas en '{out_png}'")

# ==========================================
# 5. PIPELINE DE EJECUCIÓN
# ==========================================
def main():
# # ==========================================
#     # 3. PIPELINE DE EJECUCIÓN
#     # ==========================================
#     print("Cargando CSV Global...")
#     df = pd.read_csv("dataset_global.csv")
    
#     # 1. SPLIT RIGUROSO POR SESIÓN (Evitar Fuga de Datos)
#     gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     train_idx, val_idx = next(gss.split(df, groups=df['record']))
    
#     train_df = df.iloc[train_idx]
#     val_df = df.iloc[val_idx] # Este es tu conjunto de Test/Validación
    
#     # --- NUEVA VISUALIZACIÓN DE LOS CONJUNTOS ---
#     print("\n" + "="*50)
#     print("📊 REPORTE DE DIVISIÓN DE DATOS (SPLIT 80/20 POR SESIÓN)")
#     print("="*50)
#     print(f"Total de imágenes en el CSV global: {len(df)}")
#     print(f"-> Conjunto de ENTRENAMIENTO: {len(train_df)} imágenes ({len(train_df)/len(df)*100:.1f}%)")
#     print(f"-> Conjunto de TEST/VALIDACIÓN: {len(val_df)} imágenes ({len(val_df)/len(df)*100:.1f}%)")
    
#     print("\nDistribución de clases en ENTRENAMIENTO:")
#     print(train_df['behavior'].value_counts().to_string())
    
#     print("\nDistribución de clases en TEST/VALIDACIÓN:")
#     print(val_df['behavior'].value_counts().to_string())
#     print("="*50 + "\n")

    # 2. DATA LOADERS
    print("Cargando CSVs (split por sesión: train = 1ª sesión, val = resto)...")
    train_df = pd.read_csv("dataset_train.csv")
    val_df = pd.read_csv("dataset_val.csv")
    print(f"Frames -> Train: {len(train_df)} | Val: {len(val_df)}")

    # Pesos de clase para dirección (pooled A+B, porque el flip intercambia motores).
    pooled = pd.concat([train_df["behaviorA"], train_df["behaviorB"]]).map(DIR_MAP)
    counts = pooled.value_counts().reindex([0, 1, 2]).fillna(0)
    weights = counts.sum() / (3 * counts.replace(0, np.nan))
    class_weights = torch.tensor(weights.fillna(0).values, dtype=torch.float32)
    print(f"Pesos de clase dirección {[DIR_INV[i] for i in range(3)]}: {class_weights.tolist()}")

    BATCH_SIZE = 64
    train_loader = DataLoader(AutonomousDriveDataset(train_df, is_train=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(AutonomousDriveDataset(val_df, is_train=False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MotorControlNet(lr=1e-3, lambda_speed=1.0, class_weights=class_weights)

    csv_logger = CSVLogger(save_dir="logs", name="MotorControlNet")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_dir_f1_macro', mode='max', save_top_k=1, dirpath='checkpoints/',
        filename='motorctrl-{epoch:02d}-{val_dir_f1_macro:.3f}',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=15, accelerator='gpu', devices=1, logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor], precision='16-mixed',
    )

    print("\n🚀 ENTRENANDO MotorControlNet (imagen -> dirección+velocidad por motor) 🚀")
    trainer.fit(model, train_loader, val_loader)

    # Graficado y evaluación local (sin APIs externas).
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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from PIL import Image

# ==========================================
# 1. DEFINICIÓN DEL DATASET CON AUGMENTATION
# ==========================================
class AutonomousDriveDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        
        self.class_map = {"STOP": 0, "FORWARD": 1, "BACKWARD": 2, "LEFT": 3, "RIGHT": 4}
        
        # 1. Resize estricto a la resolución nativa de MobileNetV3 (Obligatorio)
        self.resize = T.Resize((224, 224), antialias=True)
        
        # 2. Jittering Sutil (15%) - Solo lo suficiente para no memorizar la luz del laboratorio
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)
        
        # 3. Normalización estándar de ImageNet
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        
        speedA, speedB = float(row['speedA']), float(row['speedB'])
        behavior = row['behavior']

        # --- PREPROCESAMIENTO ESPACIAL GEOMÉTRICO ---
        # SIN recortes. Aplicamos el Resize estricto a TODAS las imágenes.
        img = self.resize(img)

        # --- DATA AUGMENTATION (SOLO ENTRENAMIENTO) ---
        if self.is_train:
            img = self.jitter(img)
            
            # Mirroring Estocástico (50% de probabilidad)
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                speedA, speedB = speedB, speedA
                if behavior == "LEFT": behavior = "RIGHT"
                elif behavior == "RIGHT": behavior = "LEFT"

        # Convertir a Tensor y Normalizar
        img = TF.to_tensor(img)
        img = self.normalize(img)
        
        label = self.class_map[behavior]
        
        return img, torch.tensor(label, dtype=torch.long)

# ==========================================
# 2. MODELO: PYTORCH LIGHTNING MODULE
# ==========================================
class MobileNetV3Driver(pl.LightningModule):
    def __init__(self, num_classes=5, lr=1e-4): # Learning Rate más bajo por descongelar capas profundas
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # FASE 3: Descongelamiento Híbrido SOTA
        # Congelamos bloques 0 al 10 (Detectores de líneas base de ImageNet)
        # Descongelamos bloques 11 y 12 (Semántica específica de la pista)
        for i, child in enumerate(self.backbone.features.children()):
            if i <= 10:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
                    
        self.backbone.classifier[2] = nn.Dropout(p=0.5, inplace=True)
            
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, mode=True):
        """
        Sobrescribimos el método train para PROTEGER las capas BatchNorm.
        Incluso si descongelamos convoluciones, las estadísticas de BN no deben mutar.
        """
        super().train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.requires_grad = False
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
        
    def _calculate_f1_class(self, preds, y, class_idx):
        tp = ((preds == class_idx) & (y == class_idx)).sum().float()
        fp = ((preds == class_idx) & (y != class_idx)).sum().float()
        fn = ((preds != class_idx) & (y == class_idx)).sum().float()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        
        acc = (preds == y).float().mean()
        
        f1_left = self._calculate_f1_class(preds, y, 3)
        f1_right = self._calculate_f1_class(preds, y, 4)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_LEFT', f1_left, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_RIGHT', f1_right, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Filtramos solo los parámetros que requieren gradiente para AdamW
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.lr, 
            weight_decay=1e-4
        )
        # Extendemos el Step a 7 épocas (Paciencia Optimizadora)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

# ==========================================
# 3. GRAFICADO LOCAL DEL PROGRESO (reemplaza Weights & Biases)
# ==========================================
def plot_training_progress(metrics_csv, out_png="training_progress.png"):
    """Lee el metrics.csv del CSVLogger y grafica las curvas de entrenamiento
    localmente, sin depender de ninguna API externa."""
    if not os.path.exists(metrics_csv):
        print(f"[WARN] No se encontró {metrics_csv}; no se grafica el progreso.")
        return

    m = pd.read_csv(metrics_csv)
    # El CSVLogger escribe métricas de train y val en filas distintas por época;
    # colapsamos por época promediando (cada métrica aparece una sola vez por época).
    m = m.groupby("epoch").mean(numeric_only=True).reset_index()

    panels = [
        ("Loss", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("Accuracy", [("train_acc", "Train"), ("val_acc", "Val")]),
        ("F1 por clase (Val)", [("val_f1_LEFT", "LEFT"), ("val_f1_RIGHT", "RIGHT")]),
        ("Learning Rate", [("lr-AdamW", "LR")]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Progreso de Entrenamiento — MobileNetV3 (local)", fontsize=15)
    for ax, (title, series) in zip(axes.ravel(), panels):
        plotted = False
        for col, label in series:
            if col in m.columns and m[col].notna().any():
                d = m[["epoch", col]].dropna()
                ax.plot(d["epoch"], d[col], marker="o", label=label)
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\n📈 Curvas de entrenamiento guardadas en '{out_png}' (fuente: {metrics_csv})")


# ==========================================
# 4. PIPELINE DE EJECUCIÓN
# ==========================================
def main():
    print("Cargando CSVs (split por sesión: train = 1ª sesión, val = resto)...")
    train_df = pd.read_csv("dataset_train.csv")
    val_df = pd.read_csv("dataset_val.csv")

    print(f"Frames -> Train: {len(train_df)} | Val: {len(val_df)}")

    BATCH_SIZE = 64

    train_dataset = AutonomousDriveDataset(train_df, is_train=True)
    val_dataset = AutonomousDriveDataset(val_df, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # LR bajado a 1e-4 para proteger las convoluciones 11 y 12 descongeladas
    model = MobileNetV3Driver(num_classes=5, lr=1e-4)

    # Logger LOCAL (CSV): sin Weights & Biases, sin llamadas a API externas.
    csv_logger = CSVLogger(save_dir="logs", name="MobileNetV3_Phase3_Hybrid")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1_LEFT',
        dirpath='checkpoints/',
        filename='mobilenet-sota-{epoch:02d}-{val_f1_LEFT:.2f}',
        save_top_k=1,
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=15,          # Aumentado a 15 épocas
        accelerator='gpu',
        devices=1,
        logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed'
    )

    print("\n🚀 INICIANDO ENTRENAMIENTO (FASE 3: UNFREEZE HÍBRIDO Y RESIZE NATIVO) 🚀")
    trainer.fit(model, train_loader, val_loader)

    # Graficamos el progreso nosotros mismos a partir del log local.
    plot_training_progress(os.path.join(csv_logger.log_dir, "metrics.csv"))

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
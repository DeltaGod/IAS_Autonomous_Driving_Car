import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image

# ==========================================
# 1. DEFINICIÓN DEL DATASET CON AUGMENTATION
# ==========================================
class AutonomousDriveDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        
        # Mapeo estricto de clases
        self.class_map = {"STOP": 0, "FORWARD": 1, "BACKWARD": 2, "LEFT": 3, "RIGHT": 4}
        
        # Parámetros calibrados empíricamente
        self.jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        
        speedA, speedB = float(row['speedA']), float(row['speedB'])
        behavior = row['behavior']

        # --- DATA AUGMENTATION (SOLO ENTRENAMIENTO) ---
        if self.is_train:
            # 1. Color Jittering calibrado
            img = self.jitter(img)
            
            # 2. Mirroring Estocástico (50% de probabilidad)
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                # Invertir física
                speedA, speedB = speedB, speedA
                # Invertir etiqueta
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
    def __init__(self, num_classes=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Cargar SOTA Backbone (Pre-entrenado en ImageNet)
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # FASE 1: Congelar el backbone (Feature Extractor) para evitar Catastrophic Forgetting
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
        # Reemplazar la cabeza clasificadora original (1000 clases) por las nuestras (5)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

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
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Usamos AdamW (Adam con Weight Decay correcto)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        
        # Scheduler SOTA Clásico: Reduce el LR a la décima parte cada 5 épocas
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]

# ==========================================
# 3. PIPELINE DE EJECUCIÓN
# ==========================================
def main():
    print("Cargando CSV Global...")
    df = pd.read_csv("dataset_global.csv")
    
    # 1. SPLIT RIGUROSO POR SESIÓN (Evitar Fuga de Datos)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['record']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Sesiones (Frames) -> Train: {len(train_df)} | Val: {len(val_df)}")

    # 2. DATA LOADERS (Ajusta batch_size a 32 si tu RTX 3050 Ti de 4GB se queda sin VRAM)
    BATCH_SIZE = 64
    
    train_dataset = AutonomousDriveDataset(train_df, is_train=True)
    val_dataset = AutonomousDriveDataset(val_df, is_train=False) # NUNCA aumentar validación
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 3. CONFIGURACIÓN DEL MODELO Y LOGGING
    model = MobileNetV3Driver(num_classes=5, lr=1e-3)
    
    wandb_logger = WandbLogger(project="Autonomous-Driving-ENIB", name="MobileNetV3_Phase1")
    
    # Guardar el mejor modelo basado en valid_loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='mobilenet-sota-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 4. ENTRENAMIENTO (Phase 1: Head Only)
    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator='gpu',      # Usará tu RTX 3050 Ti automáticamente
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed'    # Ahorra muchísima VRAM (Half Precision)
    )

    print("\n🚀 INICIANDO ENTRENAMIENTO (FASE 1: FEATURE EXTRACTOR CONGELADO) 🚀")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    # Necesario para el Multiprocessing de Windows/Linux con DataLoaders
    torch.set_float32_matmul_precision('medium')
    main()
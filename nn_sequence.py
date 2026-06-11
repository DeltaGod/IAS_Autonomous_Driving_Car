"""
nn_sequence.py — MOTOR de entrenamiento SECUENCIAL CNN + GRU/LSTM (no es un entrypoint).
=========================================================================================
Modelo: SECUENCIA de imágenes -> control por motor (mismas 4 salidas que nn_perframe).

POR QUÉ una recurrente: el error dominante del per-frame es STOP vs FORWARD, porque la
VELOCIDAD no está en un único frame (un auto quieto y uno avanzando despacio en el mismo
punto son idénticos). Una secuencia muestra si la escena avanza -> la GRU/LSTM resuelve
esa ambigüedad temporal.

ARQUITECTURA: MobileNetV3-small CONGELADO como extractor por-frame (TimeDistributed)
-> GRU/LSTM sobre los T vectores -> cuello compartido -> 4 cabezas. Se reutiliza toda la
lógica de loss/métricas/eval/plots de `nn_perframe` (herencia) para que la comparación
con el modelo por-frame sea justa.

Este archivo NO se corre directo: expone `run_experiment(cfg, tag)`. Cada variante RNN
vive en su `model_NN_*.py` con su `Config` congelada (ver `results/README.md`):
  model_02_gru_v1.py  model_03_gru_v2.py  model_04_lstm.py  model_05_gru_v3_antiredun.py
"""

import os
import random
import warnings

# El ajuste de hue (PIL) castea floats negativos a uint8; numpy>=1.24 lo avisa como
# "invalid value encountered in cast". Es BENIGNO: el wraparound es justo lo que el hue
# circular necesita. Lo silenciamos para no inundar el log (no cambia el resultado).
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pytorch_lightning as pl
from PIL import Image

from config import active_params_text
# Reutilizamos del modelo por-frame: mapeos, loss/métricas (herencia), eval, plots, utils.
from nn_perframe import (
    MotorControlNet, DIR_MAP, DIR_INV, SPEED_SCALE, TURNS,
    compute_class_weights, evaluate_and_plot, plot_training_progress,
    save_config_snapshot, make_trainer,
)


# ==========================================
# 1. DATASET DE SECUENCIAS (ventana many-to-one, todo desde cfg)
# ==========================================
class SequenceDriveDataset(Dataset):
    def __init__(self, df, is_train=True, cfg=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.cfg = cfg

        self.resize = T.Resize((cfg.image_size, cfg.image_size), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Ventanas válidas: una por cada frame que tenga (seq_len-1)*stride frames previos
        # DENTRO de su mismo record. En train se expanden con resample_factor por clase.
        self.samples = self._build_samples()

    def _build_samples(self):
        T_, k = self.cfg.seq_len, self.cfg.seq_stride
        # Anti-redundancia: en TRAIN subsamplea las ventanas (1 de cada wstep) para cortar
        # el solape entre secuencias casi idénticas (sliding window a 22fps). Val sin tocar.
        wstep = getattr(self.cfg, "seq_window_step", 1) if self.is_train else 1
        span = (T_ - 1) * k
        base = []
        for _, g in self.df.groupby("record", sort=False):
            idxs = g.sort_values("time_in_ms").index.to_numpy()
            for pos in range(span, len(idxs), wstep):
                win = idxs[pos - span: pos + 1: k]  # T_ índices, orden temporal ascendente
                if len(win) == T_:
                    base.append(win)
        if not self.is_train:
            return base
        # Resample por el behavior global del frame-ETIQUETA (el último de la ventana).
        rng = np.random.default_rng(self.cfg.seed)
        factors = self.cfg.resample_factor
        beh = self.df["behavior"].values
        out = []
        for win in base:
            f = float(factors.get(beh[win[-1]], 1.0))
            n = int(f)
            out.extend([win] * n)
            frac = f - n
            if frac > 0 and rng.random() < frac:
                out.append(win)
        return out

    def label_frames(self):
        """DataFrame de los frames-etiqueta (último de cada ventana, post-resample).
        Sirve para calcular class_weights sobre lo que el modelo realmente ve."""
        last = [w[-1] for w in self.samples]
        return self.df.iloc[last]

    def __len__(self):
        return len(self.samples)

    # --- color jitter CONSISTENTE en toda la secuencia (mismos factores por frame) ---
    def _sample_jitter(self):
        c = self.cfg
        b = random.uniform(max(0.0, 1 - c.jitter_brightness), 1 + c.jitter_brightness)
        co = random.uniform(max(0.0, 1 - c.jitter_contrast), 1 + c.jitter_contrast)
        s = random.uniform(max(0.0, 1 - c.jitter_saturation), 1 + c.jitter_saturation)
        h = random.uniform(-c.jitter_hue, c.jitter_hue)
        return b, co, s, h

    @staticmethod
    def _apply_jitter(img, jit):
        b, co, s, h = jit
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, co)
        img = TF.adjust_saturation(img, s)
        img = TF.adjust_hue(img, h)
        return img

    def __getitem__(self, k):
        win = self.samples[k]
        rows = self.df.iloc[win]
        label = rows.iloc[-1]
        speedA, speedB = float(label["speedA"]), float(label["speedB"])
        dirA, dirB = label["behaviorA"], label["behaviorB"]

        do_flip, jit = False, None
        if self.is_train:
            # El flip se decide UNA vez y se aplica IGUAL a toda la secuencia (+ swap A<->B).
            p = self.cfg.flip_prob_turns if label["behavior"] in TURNS else self.cfg.flip_prob_straight
            do_flip = torch.rand(1).item() < p
            if do_flip:
                speedA, speedB = speedB, speedA
                dirA, dirB = dirB, dirA
            jit = self._sample_jitter()

        frames = []
        for path in rows["image_path"]:
            img = self.resize(Image.open(path).convert("RGB"))
            if self.is_train:
                img = self._apply_jitter(img, jit)
                if do_flip:
                    img = TF.hflip(img)
            frames.append(self.normalize(TF.to_tensor(img)))
        seq = torch.stack(frames)  # (T, C, H, W)

        return seq, {
            "dirA": torch.tensor(DIR_MAP[dirA], dtype=torch.long),
            "dirB": torch.tensor(DIR_MAP[dirB], dtype=torch.long),
            "spdA": torch.tensor(speedA / SPEED_SCALE, dtype=torch.float32),
            "spdB": torch.tensor(speedB / SPEED_SCALE, dtype=torch.float32),
        }


# ==========================================
# 2. MODELO CNN + GRU/LSTM (hereda loss/métricas de MotorControlNet)
# ==========================================
class MotorControlGRU(MotorControlNet):
    """Backbone congelado por-frame -> GRU/LSTM -> cuello -> 4 cabezas.
    Reutiliza _compute_loss / training_step / validation_step / on_validation_epoch_*
    / configure_optimizers de MotorControlNet (misma loss y misma métrica F1)."""

    def __init__(self, neck_hidden=(256,), dropout=0.4, rnn_type="GRU", rnn_hidden=128,
                 rnn_layers=1, lr=1e-3, lambda_speed=1.0, weight_decay=1e-4,
                 scheduler_step=7, scheduler_gamma=0.1, class_weights=None):
        # Saltamos el __init__ de MotorControlNet (arma otra cabeza) y vamos al de Lightning.
        pl.LightningModule.__init__(self)
        # Referenciar __class__ fuerza la creación de la celda __class__ del método: sin
        # ella, save_hyperparameters de Lightning 2.x no reconoce este __init__ como el de
        # una clase y NO captura ningún hparam (quedaría hparams.lr vacío -> KeyError).
        _ = __class__  # noqa: F821
        self.save_hyperparameters("neck_hidden", "dropout", "rnn_type", "rnn_hidden",
                                  "rnn_layers", "lr", "lambda_speed", "weight_decay",
                                  "scheduler_step", "scheduler_gamma")

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        feat_dim = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()  # extractor fijo: BN usa stats de ImageNet, no se actualiza
        self.backbone = backbone

        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(input_size=feat_dim, hidden_size=rnn_hidden, num_layers=rnn_layers,
                           batch_first=True, dropout=(dropout if rnn_layers > 1 else 0.0))

        # Cuello compartido sobre el último estado oculto (igual idea que el por-frame).
        layers, d = [], rnn_hidden
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

    def forward(self, x):  # x: (B, T, C, H, W)
        B, Tlen = x.shape[0], x.shape[1]
        # Backbone congelado: sin grad y en eval (lo re-aseguramos porque trainer.train()
        # lo pondría en modo train cada época). Solo entrenan rnn + neck + cabezas.
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(x.reshape(B * Tlen, *x.shape[2:]))
        feats = feats.view(B, Tlen, -1)
        out = self.rnn(feats)[0]          # GRU y LSTM devuelven (output, hidden); tomamos output
        z = self.neck(out[:, -1])          # último timestep -> many-to-one
        return (self.head_dirA(z), self.head_dirB(z),
                torch.sigmoid(self.head_spdA(z)).squeeze(1),
                torch.sigmoid(self.head_spdB(z)).squeeze(1))


# ==========================================
# 3. PIPELINE
# ==========================================
def run_experiment(cfg, tag, results_dir="results"):
    """Entrena el modelo SECUENCIAL con la `cfg` congelada y deja TODO en results/<tag>/."""
    torch.set_float32_matmul_precision('medium')
    out = os.path.join(results_dir, tag)
    os.makedirs(out, exist_ok=True)
    save_config_snapshot(cfg, os.path.join(out, "config_used.txt"),
                         extra=f"# {tag} — modelo SECUENCIAL {cfg.rnn_type} (nn_sequence.py)")

    print(f"=== {tag} | SECUENCIAL {cfg.rnn_type} === Cargando CSVs...")
    train_df = pd.read_csv(cfg.train_csv)
    val_df = pd.read_csv(cfg.val_csv)

    train_ds = SequenceDriveDataset(train_df, is_train=True, cfg=cfg)
    val_ds = SequenceDriveDataset(val_df, is_train=False, cfg=cfg)
    print(f"Secuencias (T={cfg.seq_len}, stride={cfg.seq_stride}, wstep={cfg.seq_window_step}) -> "
          f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    if len(val_ds) < 50:
        print(f"[ADVERTENCIA] Solo {len(val_ds)} secuencias de validación: "
              f"la métrica va a ser RUIDOSA (val es corto y T/stride lo reducen más).")
    print("Parámetros activos:\n" + active_params_text(cfg))

    class_weights = compute_class_weights(train_ds.label_frames()) if cfg.use_class_weights else None
    if class_weights is not None:
        print(f"Pesos de clase {[DIR_INV[i] for i in range(3)]}: {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = MotorControlGRU(
        neck_hidden=tuple(cfg.neck_hidden), dropout=cfg.dropout,
        rnn_type=cfg.rnn_type, rnn_hidden=cfg.rnn_hidden, rnn_layers=cfg.rnn_layers,
        lr=cfg.lr, lambda_speed=cfg.lambda_speed, weight_decay=cfg.weight_decay,
        scheduler_step=cfg.scheduler_step, scheduler_gamma=cfg.scheduler_gamma,
        class_weights=class_weights,
    )

    trainer, csv_logger, checkpoint_callback = make_trainer(cfg, out, ckpt_prefix="motorgru")
    print(f"\n🚀 ENTRENANDO {tag} (MotorControlGRU / {cfg.rnn_type}) 🚀")
    trainer.fit(model, train_loader, val_loader)

    plot_training_progress(os.path.join(csv_logger.log_dir, "metrics.csv"),
                           out_png=os.path.join(out, "training_progress.png"), cfg=cfg)
    best = checkpoint_callback.best_model_path
    if best and os.path.exists(best):
        print(f"\nEvaluando mejor checkpoint: {best}")
        model = MotorControlGRU.load_from_checkpoint(best, class_weights=class_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_and_plot(model, val_loader, device,
                      out_png=os.path.join(out, "eval_confusion.png"),
                      report_path=os.path.join(out, "report.txt"), cfg=cfg)
    print(f"\n✅ {tag} listo. Resultados en {out}/")

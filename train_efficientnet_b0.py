# train_efficientnet_b0.py

# ------------------------------------------------------------

# EfficientNet-B0 UNet training script for EO-SAR change detection

# Uses segmentation_models_pytorch (SMP)

# Designed as a lightweight improved-generalization experiment

# against the ResNet18 baseline.

# ------------------------------------------------------------

# INSTALL:

# pip install segmentation-models-pytorch timm

# ------------------------------------------------------------

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time

from data.dataset import CachedEOSARDataset
from losses.combined import CombinedLoss
from utils.metrics import ChangeMetrics

# ============================================================

# CONFIG

# ============================================================

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# ============================================================

# DEVICE

# ============================================================

device = (
torch.device("cuda") if torch.cuda.is_available()
else torch.device("cpu")
)

print(f"Device: {device}")

# ============================================================

# DATALOADERS

# ============================================================

print("\nBuilding dataloaders...")

train_ds = CachedEOSARDataset(
cfg["data"]["cache_dir"],
split="train",
augment=True,
oversample_change=cfg["data"]["oversample_factor"],
)

val_ds = CachedEOSARDataset(
cfg["data"]["cache_dir"],
split="val",
augment=False,
oversample_change=0,
)

train_loader = DataLoader(
train_ds,
batch_size=cfg["training"]["batch_size"],
shuffle=True,
num_workers=cfg["data"]["num_workers"],
pin_memory=cfg["data"]["pin_memory"],
)

val_loader = DataLoader(
val_ds,
batch_size=cfg["evaluation"]["batch_size"],
shuffle=False,
num_workers=cfg["data"]["num_workers"],
pin_memory=cfg["data"]["pin_memory"],
)

# ============================================================

# MODEL

# ============================================================

print("\nBuilding EfficientNet-B0 UNet...")

model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=5,
    classes=1,
    decoder_channels=(128, 64, 32, 16, 8),
).to(device)

# Extra dropout for scene generalization
model.segmentation_head = nn.Sequential(
    nn.Dropout2d(0.10),
    model.segmentation_head
)

# ============================================================

# OPTIONAL: FREEZE EARLY ENCODER BLOCKS

# ============================================================

# Helps:

# - reduce CPU load

# - reduce overfitting

# - improve transfer stability

# ------------------------------------------------------------

freeze_encoder = False

if freeze_encoder:
    encoder_children = list(model.encoder.children())
    n_freeze = int(len(encoder_children) * 0.40)
    for child in encoder_children[:n_freeze]:
        for param in child.parameters():
            param.requires_grad = False
    print(f"Frozen {n_freeze}/{len(encoder_children)} "
          f"encoder blocks")


# ============================================================

# PARAM COUNT

# ============================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
p.numel() for p in model.parameters()
if p.requires_grad
)

print(f"Total params:     {total_params/1e6:.2f}M")
print(f"Trainable params: {trainable_params/1e6:.2f}M")

# ============================================================

# LOSS

# ============================================================

criterion = CombinedLoss(cfg)

# ============================================================

# OPTIMIZER

# ============================================================

optimizer = optim.AdamW([
    {"params": filter(
        lambda p: p.requires_grad,
        model.encoder.parameters()),
     "lr": 5e-5},                      # encoder: smaller LR
    {"params": model.decoder.parameters(),
     "lr": 1.5e-4},                      # decoder: normal LR
    {"params": model.segmentation_head.parameters(),
     "lr": 1.5e-4},
], weight_decay=cfg["optimizer"]["weight_decay"])   

# ============================================================

# SCHEDULER

# ============================================================
EPOCHS        = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# ============================================================

# OUTPUT DIRS

# ============================================================

SAVE_DIR = Path("checkpoints/efficientnet_b0")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("logs/efficientnet_b0")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = SAVE_DIR / "best_model.pth"

# ============================================================

# TRAIN LOOP

# ============================================================



def train_epoch(model, loader):
    model.train()

    metrics = ChangeMetrics(
        threshold=cfg["training"]["threshold"]
    )

    total_loss = 0.0

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        leave=False,
        desc="Training",
        dynamic_ncols=True
    )

    for i, batch in pbar:

        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        vm  = batch["valid_mask"].to(device)

        optimizer.zero_grad()

        out = model(inp)

        loss, _ = criterion(out, tgt, vm)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["optimizer"]["grad_clip"]
        )

        optimizer.step()

        metrics.update(out.detach(), tgt, vm)

        total_loss += loss.item()

        res = metrics.compute()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "f1": f"{res['f1']:.3f}"
        })

    return total_loss / len(loader), metrics.compute()
# ============================================================

# VALIDATION LOOP

# ============================================================

# ============================================================
# VALIDATION LOOP
# ============================================================

def validate(model, loader):
    model.eval()

    metrics = ChangeMetrics(
        threshold=cfg["training"]["threshold"]
    )

    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            vm  = batch["valid_mask"].to(device)

            out = model(inp)

            loss, _ = criterion(out, tgt, vm)

            metrics.update(out, tgt, vm)

            total_loss += loss.item()

    results = metrics.compute()

    return total_loss / len(loader), results
# ============================================================

# HISTORY

# ============================================================

history = {
"train_loss": [],
"val_loss": [],
"train_f1": [],
"val_f1": [],
"train_iou": [],
"val_iou": [],
}


# ============================================================

# TRAINING

# ============================================================

print("\n" + "="*60)
print("TRAINING — EfficientNet-B0 UNet")
print("="*60)


patience_limit= 5
best_f1       = 0.0
patience      = 0

for epoch in range(EPOCHS):
    lr = optimizer.param_groups[0]["lr"]
    print(f"\nEpoch {epoch+1}/{EPOCHS}  (LR={lr:.2e})")
    epoch_start = time.time()
    # Unfreeze encoder after epoch 5
    if epoch == 5 and freeze_encoder:
        print("  Unfreezing full encoder...")
        for param in model.encoder.parameters():
            param.requires_grad = True
        # Reduce encoder LR further after unfreezing
        optimizer.param_groups[0]["lr"] = 5e-6

    train_loss, train_res = train_epoch(model, train_loader)

    val_loss, val_res = validate(model, val_loader)

    # Scheduler steps AFTER optimizer
    scheduler.step()

    print(f"  [Train] loss={train_loss:.4f}  "
          f"F1={train_res['f1']:.4f}  "
          f"IoU={train_res['iou']:.4f}")
    print(f"  [Val]   loss={val_loss:.4f}  "
          f"F1={val_res['f1']:.4f}  "
          f"IoU={val_res['iou']:.4f}  "
          f"P={val_res['precision']:.4f}  "
          f"R={val_res['recall']:.4f}")
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_f1"].append(train_res["f1"])
    history["val_f1"].append(val_res["f1"])

    history["train_iou"].append(train_res["iou"])
    history["val_iou"].append(val_res["iou"])
    epoch_time = (time.time() - epoch_start) / 60

    remaining = epoch_time * (EPOCHS - epoch - 1)

    print(
        f"  Time: {epoch_time:.1f} min/epoch | "
        f"ETA: {remaining:.1f} min remaining"
)
    if val_res["f1"] > best_f1:
        best_f1  = val_res["f1"]
        patience = 0
        torch.save({
            "epoch":       epoch + 1,
            "model_state": model.state_dict(),
            "val_f1":      best_f1,
            "val_iou":     val_res["iou"],
        }, BEST_PATH)
        print(f"  ✅ Saved best  F1={best_f1:.4f}")
    else:
        patience += 1
        print(f"  No improvement ({patience}/{patience_limit})")
        if patience >= patience_limit:
            print("Early stopping.")
            break


# ============================================================

# SAVE HISTORY

# ============================================================

with open(LOG_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nTraining complete.")
print(f"Best validation F1: {best_f1:.4f}")
print(f"Best model saved to: {BEST_PATH}")

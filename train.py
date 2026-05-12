"""
train.py — EO-SAR Change Detection Training Script

Usage:
    python train.py --config config.yaml

Outputs:
    checkpoints/best_model.pth
    logs/training_history.json
""" 


import json
import time
import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path

from data.dataset      import build_loaders
from models.change_detector import build_model
from losses.combined   import CombinedLoss
from utils.metrics     import ChangeMetrics, tune_threshold


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device


# ------------------------------------------------------------------
# LR schedule: linear warmup + cosine annealing
# ------------------------------------------------------------------

def build_scheduler(optimizer, cfg):
    warmup = cfg["scheduler"]["warmup_epochs"]
    epochs = cfg["training"]["epochs"]
    eta    = cfg["scheduler"]["eta_min"]
    lr     = cfg["optimizer"]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return eta / lr + (1 - eta / lr) * 0.5 * (
            1 + np.cos(np.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Train one epoch
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    metrics    = ChangeMetrics(threshold=0.5)
    total_loss = 0.0
    n          = 0
    t0         = time.time()

    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        vm  = batch["valid_mask"].to(device)

        optimizer.zero_grad()
        logits = model(inp)
        loss, comps = criterion(logits, tgt, vm)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm = cfg_global["optimizer"]["grad_clip"]
        )
        optimizer.step()

        total_loss += comps["total"]
        metrics.update(logits, tgt, vm)
        n += 1

    res     = metrics.compute()
    elapsed = time.time() - t0
    print(f"  [Train] loss={total_loss/n:.4f}  "
          f"F1={res['f1']:.4f}  IoU={res['iou']:.4f}  "
          f"P={res['precision']:.4f}  R={res['recall']:.4f}  "
          f"({elapsed:.0f}s)")
    return total_loss / n, res


# ------------------------------------------------------------------
# Validate one epoch
# ------------------------------------------------------------------

@torch.no_grad()
def val_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    metrics    = ChangeMetrics(threshold=threshold)
    total_loss = 0.0
    n          = 0

    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        vm  = batch["valid_mask"].to(device)

        logits = model(inp)
        loss, _ = criterion(logits, tgt, vm)
        total_loss += loss.item()
        metrics.update(logits, tgt, vm)
        n += 1

    res = metrics.compute()
    print(f"  [Val]   loss={total_loss/n:.4f}  "
          f"F1={res['f1']:.4f}  IoU={res['iou']:.4f}  "
          f"P={res['precision']:.4f}  R={res['recall']:.4f}")
    return total_loss / n, res


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

cfg_global = {}   # accessible in train_epoch for grad_clip

def main(cfg):
    global cfg_global
    cfg_global = cfg

    set_seed(cfg["project"]["seed"])
    device = get_device()

    # Data
    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader = build_loaders(cfg)

    # Model
    print("\nBuilding model...")
    model = build_model(cfg).to(device)

    # Loss
    criterion = CombinedLoss(cfg).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["optimizer"]["lr"],
        weight_decay = cfg["optimizer"]["weight_decay"],
    )

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg)

    # Backbone-specific output dirs
    backbone   = cfg["model"]["backbone"]
    ckpt_dir   = Path(cfg["outputs"]["checkpoints"]) / backbone
    log_dir    = Path(cfg["outputs"]["logs"])         / backbone
    result_dir = Path(cfg["outputs"]["results"])      / backbone
    gcam_dir   = Path(cfg["outputs"]["gradcam"])      / backbone

    for d in [ckpt_dir, log_dir, result_dir, gcam_dir]:
        d.mkdir(parents=True, exist_ok=True)

    save_path = ckpt_dir / "best_model.pth"
    best_f1    = 0.0
    start_epoch = 1

    # ------------------------------------------------------------
    # Resume from checkpoint if exists
    # ------------------------------------------------------------

    if save_path.exists():

        print(f"\n🔄 Resuming from checkpoint: {save_path}")

        ckpt = torch.load(save_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])

        best_f1 = ckpt.get("val_f1", 0.0)

        start_epoch = ckpt["epoch"] + 1

        # Advance scheduler to correct epoch
        # for _ in range(start_epoch - 1):
        #     scheduler.step()
        # Restore scheduler epoch
        scheduler.last_epoch = start_epoch - 2
        scheduler.step()
        print(
            f"  Resumed from epoch {ckpt['epoch']} "
            f"(best F1={best_f1:.4f})"
        )

    print(f"Checkpoints → {ckpt_dir}")
    print(f"Logs        → {log_dir}")
    print(f"Results     → {result_dir}")

    # Training state
    history    = {k: [] for k in [
        "train_loss", "val_loss",
        "train_f1",   "val_f1",
        "train_iou",  "val_iou",
    ]}

    no_improve = 0
    epochs     = cfg["training"]["epochs"]
    patience   = cfg["training"]["patience"]
    threshold  = cfg["training"]["threshold"]

    print(f"\n{'='*60}")
    print(f"TRAINING — {cfg['model']['backbone'].upper()}")
    print(f"  LR={cfg['optimizer']['lr']}  "
          f"pos_weight={cfg['loss']['pos_weight']}  "
          f"Epochs={epochs}  Batch={cfg['training']['batch_size']}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, epochs + 1):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{epochs}  (LR={current_lr:.2e})")

        tr_loss, tr_res = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        vl_loss, vl_res = val_epoch(
            model, val_loader, criterion, device, threshold
        )
        scheduler.step()

        # Log history
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_res["f1"])
        history["val_f1"].append(vl_res["f1"])
        history["train_iou"].append(tr_res["iou"])
        history["val_iou"].append(vl_res["iou"])

        # Save best model
        if vl_res["f1"] > best_f1:
            best_f1 = vl_res["f1"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_f1":      best_f1,
                "val_iou":     vl_res["iou"],
                "config":      cfg,
            }, save_path)
            print(f"  ✅ Saved best model  F1={best_f1:.4f}  "
                  f"IoU={vl_res['iou']:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break

    # Save history
    json.dump(history,
              open(log_dir / "training_history.json", "w"),
              indent=2)
    print(f"\nBest Val F1: {best_f1:.4f}")
    print(f"Checkpoint:  {save_path}")

    # Threshold tuning on val
    print("\n" + "="*60)
    print("THRESHOLD TUNING ON VALIDATION SET")
    print("="*60)
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    best_thresh, _ = tune_threshold(model, val_loader, device)

    # Final eval on both splits
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    for split_name, loader in [("VAL", val_loader), ("TEST", test_loader)]:
        metrics = ChangeMetrics(threshold=best_thresh)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inp = batch["input"].to(device)
                tgt = batch["target"].to(device)
                vm  = batch["valid_mask"].to(device)
                out = model(inp)
                metrics.update(out, tgt, vm)
        print(f"\n{split_name} (threshold={best_thresh}):")
        metrics.print_results(split_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
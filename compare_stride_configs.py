# compare_stride_configs.py
# ------------------------------------------------------------
# Compare:
#   stride=128 vs stride=192 vs stride=256
#
# Measures:
#   - Training time
#   - Epoch time
#   - Val F1 / IoU / Precision / Recall
#   - Learning speed
#
# Usage:
#   python compare_stride_configs.py
#
# Assumes:
#   data/cache_256_s128
#   data/cache_256_s192
#   data/cache_256_s256
# already exist.
# ------------------------------------------------------------

import os
import time
import json
import copy
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from torch.utils.data import DataLoader

from data.dataset import CachedEOSARDataset
from models.change_detector import build_model
from losses.combined import CombinedLoss
from utils.metrics import ChangeMetrics


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = "config.yaml"

CACHE_CONFIGS = {
    "stride128": "data/cache_256_s128",
    "stride192": "data/cache_256_s192",
    "stride256": "data/cache_256_s256",
}

EPOCHS = 3               # short comparison run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Smaller for fast comparison
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE   = 4

RESULTS_FILE = "stride_comparison_results.json"


# ============================================================
# HELPERS
# ============================================================

def build_loaders(cache_dir):

    train_ds = CachedEOSARDataset(
        cache_dir=cache_dir,
        split="train",
        augment=True,
        oversample_change=True
    )

    val_ds = CachedEOSARDataset(
        cache_dir=cache_dir,
        split="val",
        augment=False,
        oversample_change=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    metrics = ChangeMetrics(threshold=0.45)

    total_loss = 0.0

    start = time.time()

    for i, batch in enumerate(loader):

        if i >= 100:
            break

        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        vm  = batch["valid_mask"].to(device)

        optimizer.zero_grad()

        out = model(inp)

        loss,loss_dict = criterion(out, tgt, vm)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        metrics.update(out.detach(), tgt, vm)

    elapsed = time.time() - start

    res = metrics.compute()

    return {
        "loss": total_loss / i+1,
        "f1": res["f1"],
        "iou": res["iou"],
        "precision": res["precision"],
        "recall": res["recall"],
        "time_sec": elapsed
    }


@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    metrics = ChangeMetrics(threshold=0.45)

    total_loss = 0.0

    start = time.time()

    for i, batch in enumerate(loader):

        if i >= 300:
            break
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        vm  = batch["valid_mask"].to(device)

        out = model(inp)

        loss,loss_dict = criterion(out, tgt, vm)

        total_loss += loss.item()

        metrics.update(out, tgt, vm)

    elapsed = time.time() - start

    res = metrics.compute()

    return {
        "loss": total_loss / len(loader),
        "f1": res["f1"],
        "iou": res["iou"],
        "precision": res["precision"],
        "recall": res["recall"],
        "time_sec": elapsed
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STRIDE COMPARISON EXPERIMENT")
    print("=" * 70)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    all_results = {}

    for stride_name, cache_dir in CACHE_CONFIGS.items():

        print("\n" + "=" * 70)
        print(f"TESTING: {stride_name}")
        print("=" * 70)

        train_loader, val_loader = build_loaders(cache_dir)

        model = build_model(cfg).to(DEVICE)

        # criterion = CombinedLoss(
        #     bce_weight=cfg["loss"]["bce_weight"],
        #     dice_weight=cfg["loss"]["dice_weight"],
        #     focal_weight=cfg["loss"]["focal_weight"],
        #     pos_weight=cfg["loss"]["pos_weight"],
        #     focal_alpha=cfg["loss"]["focal_alpha"],
        #     focal_gamma=cfg["loss"]["focal_gamma"]
        # )
        criterion = CombinedLoss(cfg)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"]
        )

        stride_results = []

        total_start = time.time()

        best_f1 = -1

        for epoch in range(EPOCHS):

            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            train_res = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                DEVICE
            )

            val_res = validate(
                model,
                val_loader,
                criterion,
                DEVICE
            )

            best_f1 = max(best_f1, val_res["f1"])

            epoch_result = {
                "epoch": epoch + 1,
                "train": train_res,
                "val": val_res
            }

            stride_results.append(epoch_result)

            print(
                f"  Train: "
                f"loss={train_res['loss']:.4f} "
                f"F1={train_res['f1']:.4f} "
                f"IoU={train_res['iou']:.4f} "
                f"time={train_res['time_sec']/60:.1f}m"
            )

            print(
                f"  Val:   "
                f"loss={val_res['loss']:.4f} "
                f"F1={val_res['f1']:.4f} "
                f"IoU={val_res['iou']:.4f} "
                f"P={val_res['precision']:.4f} "
                f"R={val_res['recall']:.4f} "
                f"time={val_res['time_sec']/60:.1f}m"
            )

        total_time = time.time() - total_start

        all_results[stride_name] = {
            "cache_dir": cache_dir,
            "epochs": stride_results,
            "best_f1": best_f1,
            "total_time_hours": total_time / 3600,
            "avg_epoch_minutes": total_time / EPOCHS / 60,
            "n_train_batches": len(train_loader),
            "n_val_batches": len(val_loader),
        }

        print("\nSUMMARY")
        print(f"  Best F1:          {best_f1:.4f}")
        print(f"  Total time:       {total_time/3600:.2f} hr")
        print(f"  Avg epoch time:   {total_time/EPOCHS/60:.1f} min")
        print(f"  Train batches:    {len(train_loader)}")

    # ========================================================
    # FINAL COMPARISON TABLE
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for name, res in all_results.items():

        print(f"\n{name}")
        print(f"  Best F1:          {res['best_f1']:.4f}")
        print(f"  Avg Epoch Time:   {res['avg_epoch_minutes']:.1f} min")
        print(f"  Train Batches:    {res['n_train_batches']}")

        f1s = [e["val"]["f1"] for e in res["epochs"]]

        if len(f1s) >= 2:
            learning_gain = f1s[-1] - f1s[0]
        else:
            learning_gain = 0

        print(f"  Learning Gain:    {learning_gain:+.4f}")

    # Save JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Saved results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
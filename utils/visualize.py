"""
utils/visualize.py — Visualization Utilities

Functions:
    plot_training_curves    : loss, F1, IoU over epochs
    plot_confusion_matrix   : seaborn heatmap
    plot_qualitative        : success + failure prediction examples
    plot_gradcam            : GradCAM++ and LayerCAM side by side
    plot_sar_channels       : raw / CLAHE / log SAR comparison
    plot_batch_sample       : debug a single dataloader batch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


def _norm(arr):
    """Normalize array to 0-1 for display."""
    arr = arr.astype(float)
    mn, mx = arr.min(), arr.max()
    return np.clip((arr - mn) / (mx - mn + 1e-8), 0, 1)


def _eo_rgb(tensor_chw):
    """Convert (3, H, W) EO tensor to (H, W, 3) display image."""
    img = tensor_chw[:3].permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)


# ------------------------------------------------------------------

def plot_training_curves(history, save_dir="results"):
    """
    Plot loss, F1, IoU training curves.

    history: dict with keys train_loss, val_loss,
             train_f1, val_f1, train_iou, val_iou
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("Loss",     "train_loss", "val_loss"),
        ("F1 Score", "train_f1",   "val_f1"),
        ("IoU",      "train_iou",  "val_iou"),
    ]

    for ax, (title, tr_key, vl_key) in zip(axes, pairs):
        ax.plot(history[tr_key], label="Train", linewidth=2)
        ax.plot(history[vl_key], label="Val",   linewidth=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = save_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------

def plot_confusion_matrix(cm, split="test", save_dir="results"):
    """
    Plot confusion matrix heatmap.

    cm: (2, 2) numpy array [[TN, FP], [FN, TP]]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred NoChange", "Pred Change"],
        yticklabels=["GT NoChange",   "GT Change"],
        ax=ax, annot_kws={"size": 13},
    )
    ax.set_title(f"{split.upper()} Confusion Matrix", fontsize=13)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Prediction")
    plt.tight_layout()
    path = save_dir / f"confusion_matrix_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------

def plot_qualitative(cases, labels, split="test",
                     threshold=0.45, save_dir="results"):
    """
    Plot qualitative prediction results.

    cases: list of dicts with keys:
        eo   : (H, W, 3) float 0-1
        sar  : (H, W)    float 0-1
        gt   : (H, W)    int   0/1
        pred : (H, W)    int   0/1
        iou  : float

    labels: list of strings e.g. ["[OK] IoU=0.67", "[FAIL] IoU=0.01"]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if not cases:
        print("[WARN]  No cases to plot")
        return

    n   = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["EO Pre-Event", "SAR Post (CLAHE)",
                  "Ground Truth", f"Prediction (t={threshold})"]

    for i, (case, lbl) in enumerate(zip(cases, labels)):
        row = axes[i]

        row[0].imshow(np.clip(case["eo"], 0, 1))
        row[0].set_title(f"{lbl}\n{col_titles[0]}", fontsize=9)

        row[1].imshow(case["sar"], cmap="gray")
        row[1].set_title(col_titles[1], fontsize=9)

        row[2].imshow(case["gt"],   cmap="Reds", vmin=0, vmax=1)
        row[2].set_title(col_titles[2], fontsize=9)

        row[3].imshow(case["pred"], cmap="Reds", vmin=0, vmax=1)
        row[3].set_title(col_titles[3], fontsize=9)

        for ax in row:
            ax.axis("off")

    plt.suptitle(
        f"Qualitative Results — {split.upper()}\n"
        f"[OK] Success (IoU>0.4)   [FAIL] Failure (IoU<0.15)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = save_dir / f"qualitative_{split}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------

def plot_gradcam(eo, sar, gt, pred, cam_pp, cam_lc,
                 leak_pp, leak_lc, threshold=0.45,
                 split="test", save_dir="results"):
    """
    Plot GradCAM++ and LayerCAM side by side.

    eo:     (H, W, 3) float 0-1
    sar:    (H, W)    float 0-1
    gt:     (H, W)    int 0/1
    pred:   (H, W)    int 0/1
    cam_pp: (H, W)    float 0-1  GradCAM++
    cam_lc: (H, W)    float 0-1  LayerCAM
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # Row 0: raw inputs + prediction
    axes[0][0].imshow(np.clip(eo, 0, 1))
    axes[0][0].set_title("EO Pre-Event")

    axes[0][1].imshow(sar, cmap="gray")
    axes[0][1].set_title("SAR Post (CLAHE)")

    axes[0][2].imshow(gt, cmap="Reds", vmin=0, vmax=1)
    axes[0][2].set_title("Ground Truth")

    axes[0][3].imshow(pred, cmap="Reds", vmin=0, vmax=1)
    axes[0][3].set_title(f"Prediction (t={threshold})")

    # Row 1: GradCAM overlays
    axes[1][0].imshow(np.clip(eo, 0, 1))
    axes[1][0].imshow(cam_pp, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1][0].set_title(
        f"GradCAM++ overlay\nleak={leak_pp:.3f} "
        f"{'[OK]' if leak_pp < 0.05 else '[WARN]'}"
    )

    axes[1][1].imshow(cam_pp, cmap="hot")
    axes[1][1].set_title("GradCAM++ heatmap")

    axes[1][2].imshow(np.clip(eo, 0, 1))
    axes[1][2].imshow(cam_lc, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1][2].set_title(
        f"LayerCAM overlay\nleak={leak_lc:.3f} "
        f"{'[OK]' if leak_lc < 0.05 else '[WARN]'}"
    )

    axes[1][3].imshow(cam_lc, cmap="hot")
    axes[1][3].set_title("LayerCAM heatmap")

    for row in axes:
        for ax in row:
            ax.axis("off")

    plt.suptitle(
        f"GradCAM Analysis — {split.upper()}\n"
        f"leak < 0.05 means model does not activate on nodata regions",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = save_dir / f"gradcam_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------

def plot_sar_channels(sar_raw, sar_clahe, sar_log,
                      save_dir="results"):
    """
    Compare SAR raw / CLAHE / log channels.
    Used in EDA and report.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sar_raw,   cmap="gray")
    axes[0].set_title(f"SAR Raw\n"
                      f"range: {sar_raw.min():.0f}-{sar_raw.max():.0f}")

    axes[1].imshow(sar_clahe, cmap="gray")
    axes[1].set_title(f"SAR CLAHE\n"
                      f"range: {sar_clahe.min():.0f}-{sar_clahe.max():.0f}")

    axes[2].imshow(sar_log,   cmap="gray")
    axes[2].set_title(f"SAR Log\n"
                      f"range: {sar_log.min():.0f}-{sar_log.max():.0f}")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("SAR Channel Comparison", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    path = save_dir / "sar_channels.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------

def plot_batch_sample(batch, save_dir="results", tag="train"):
    """
    Debug visualization of a single dataloader batch.
    Shows EO, SAR CLAHE, SAR Log, target, valid mask.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    B   = min(4, batch["input"].shape[0])
    fig, axes = plt.subplots(B, 5, figsize=(20, 4.5 * B))
    if B == 1:
        axes = [axes]

    col_titles = ["EO RGB", "SAR CLAHE (ch3)",
                  "SAR Log (ch4)", "Target", "Valid Mask"]

    for i in range(B):
        inp = batch["input"][i]   # (5, H, W)
        tgt = batch["target"][i].numpy()
        vm  = batch["valid_mask"][i].numpy()

        eo      = np.clip(inp[:3].permute(1, 2, 0).numpy(), 0, 1)
        clahe   = inp[3].numpy()
        log_sar = inp[4].numpy()

        change_pct = 100 * tgt.mean()

        axes[i][0].imshow(eo)
        axes[i][0].set_title(f"{col_titles[0]} [{i}]", fontsize=9)

        axes[i][1].imshow(clahe, cmap="gray")
        axes[i][1].set_title(col_titles[1], fontsize=9)

        axes[i][2].imshow(log_sar, cmap="gray")
        axes[i][2].set_title(col_titles[2], fontsize=9)

        axes[i][3].imshow(tgt, cmap="Reds", vmin=0, vmax=1)
        axes[i][3].set_title(
            f"{col_titles[3]}\nchange={change_pct:.1f}%", fontsize=9
        )

        axes[i][4].imshow(vm, cmap="Greens")
        axes[i][4].set_title(
            f"{col_titles[4]}\nvalid={vm.mean():.1%}", fontsize=9
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(f"Batch Sample — {tag}", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    path = save_dir / f"batch_sample_{tag}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
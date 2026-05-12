"""
eval.py — EO-SAR Change Detection Evaluation Script

Usage:
    python eval.py --config config.yaml \
                   --weights checkpoints/best_model.pth \
                   --split test \
                   --threshold 0.45 \
                   --gradcam

Outputs:
    results/confusion_matrix_{split}.png
    results/qualitative_{split}.png
    results/gradcam_{split}.png
    results/metrics_{split}.json
"""

import argparse
import json
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader

import cv2
import torch.nn.functional as F

from data.dataset           import EOSARDataset
from models.change_detector import build_model
from utils.metrics          import ChangeMetrics
from utils.gradcam          import build_gradcam, boundary_leak_check


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    metrics = ChangeMetrics(threshold=threshold)
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        vm  = batch["valid_mask"].to(device)
        # out = model(inp)

        out1 = model(inp)

        out2 = torch.flip(
            model(torch.flip(inp, dims=[3])),
            dims=[3]
        )

        out3 = torch.flip(
            model(torch.flip(inp, dims=[2])),
            dims=[2]
        )

        out = (out1 + out2 + out3) / 3.0
        metrics.update(out, tgt, vm)
    return metrics


# ------------------------------------------------------------------
# Confusion matrix plot
# ------------------------------------------------------------------

def plot_confusion_matrix(cm, split, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred NoChange", "Pred Change"],
        yticklabels=["GT NoChange",   "GT Change"],
        ax=ax,
    )
    ax.set_title(f"{split.upper()} Confusion Matrix")
    plt.tight_layout()
    path = save_dir / f"confusion_matrix_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------
# Qualitative results
# ------------------------------------------------------------------

def plot_qualitative(model, loader, device, threshold, split, save_dir,
                     n_success=5, n_failure=3):
    model.eval()
    success, failure = [], []

    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            tgt = batch["target"]
            vm  = batch["valid_mask"]
            out1 = model(inp)

            out2 = torch.flip(
                model(torch.flip(inp, dims=[3])),
                dims=[3]
            )

            out3 = torch.flip(
                model(torch.flip(inp, dims=[2])),
                dims=[2]
            )

            out = (out1 + out2 + out3) / 3.0
            # out = model(inp)
            # pred = (out.sigmoid().squeeze(1).cpu() >= threshold).long()
            pred = (out.sigmoid().squeeze(1).cpu() >= threshold)

            # Morphological cleanup
            # pred_clean = []

            # for p in pred:
            #     p_np = p.numpy().astype(np.uint8)

            #     p_np = cv2.morphologyEx(
            #         p_np,
            #         cv2.MORPH_OPEN,
            #         np.ones((3,3), np.uint8)
            #     )

            #     pred_clean.append(torch.from_numpy(p_np))

            # pred = torch.stack(pred_clean)
            for i in range(inp.shape[0]):
                gt = tgt[i].numpy()
                pr = pred[i].numpy()
                v  = vm[i].numpy()

                if gt.sum() == 0:
                    continue

                inter      = ((pr == 1) & (gt == 1) & v).sum()
                union      = (((pr == 1) | (gt == 1)) & v).sum()
                iou_sample = inter / (union + 1e-8)

                item = {
                    "eo":   inp[i, :3].permute(1, 2, 0).cpu().numpy(),
                    "sar":  inp[i,  3].cpu().numpy(),
                    "gt":   gt,
                    "pred": pr,
                    "iou":  float(iou_sample),
                }

                if iou_sample > 0.4 and len(success) < n_success:
                    success.append(item)
                elif iou_sample < 0.15 and len(failure) < n_failure:
                    failure.append(item)

            if len(success) >= n_success and len(failure) >= n_failure:
                break

    all_cases = success + failure
    labels    = (
        [f"[OK] IoU={c['iou']:.2f}" for c in success] +
        [f"[FAIL] IoU={c['iou']:.2f}" for c in failure]
    )

    if not all_cases:
        print("⚠️  No qualifying cases found for qualitative plot")
        return

    fig, axes = plt.subplots(len(all_cases), 4,
                             figsize=(16, 4.5 * len(all_cases)))
    if len(all_cases) == 1:
        axes = [axes]

    for i, (case, lbl) in enumerate(zip(all_cases, labels)):
        axes[i][0].imshow(np.clip(case["eo"], 0, 1))
        axes[i][0].set_title(f"{lbl}\nEO Pre"); axes[i][0].axis("off")

        axes[i][1].imshow(case["sar"], cmap="gray")
        axes[i][1].set_title("SAR Post (CLAHE)"); axes[i][1].axis("off")

        axes[i][2].imshow(case["gt"],   cmap="Reds", vmin=0, vmax=1)
        axes[i][2].set_title("Ground Truth"); axes[i][2].axis("off")

        axes[i][3].imshow(case["pred"], cmap="Reds", vmin=0, vmax=1)
        axes[i][3].set_title(f"Prediction\n(thresh={threshold})"); axes[i][3].axis("off")

    plt.suptitle(f"Qualitative Results — {split.upper()}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_dir / f"qualitative_{split}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------------
# GradCAM visualization
# ------------------------------------------------------------------

def run_gradcam(model, loader, device, cfg, threshold, split, save_dir):
    from utils.gradcam import build_gradcam, boundary_leak_check
    
    
    cam_method = build_gradcam(model, cfg)
    model.eval()

    # Find a batch with change pixels
    target_batch = None
    for batch in loader:
        if (batch["target"] == 1).any():
            target_batch = batch
            break

    if target_batch is None:
        print("⚠️  No change pixels found in loader for GradCAM")
        cam_method.remove_hooks()
        return

    inp = target_batch["input"][:1].to(device)
    tgt = target_batch["target"][0].squeeze().numpy()
    vm  = target_batch["valid_mask"][0].numpy()

    # Generate both CAM methods
    cam_pp = cam_method.generate(inp)

    # LayerCAM as secondary
    from utils.gradcam import LayerCAM
    cam_layer_obj = LayerCAM(model, model.gradcam_layer)
    cam_lc = cam_layer_obj.generate(inp)
    cam_layer_obj.remove_hooks()

    # Boundary leak checks
    leak_pp = boundary_leak_check(cam_pp, vm, threshold=0.5)
    leak_lc = boundary_leak_check(cam_lc, vm, threshold=0.5)

    print(f"GradCAM++ leak: {leak_pp:.4f}")
    print(f"LayerCAM  leak: {leak_lc:.4f}")
    for name, leak in [("GradCAM++", leak_pp), ("LayerCAM", leak_lc)]:
        status = "[OK]" if leak < 0.05 else "⚠️"
        print(f"  {status} {name} leak={leak:.3f}")

    # Prediction
    with torch.no_grad():
        # pred = (model(inp).sigmoid().squeeze().cpu().numpy() >= threshold)
        pred = (
            model(inp)
            .sigmoid()
            .squeeze()
            .cpu()
            .numpy()
            >= threshold
        ).astype(np.uint8)

        pred = np.squeeze(pred)

        # pred = cv2.morphologyEx(
        #     pred,
        #     cv2.MORPH_OPEN,
        #     np.ones((3,3), np.uint8)
        # )
    eo  = np.clip(inp[0, :3].permute(1, 2, 0).cpu().numpy(), 0, 1)
    sar = inp[0, 3].cpu().numpy()

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    axes[0].imshow(eo)
    axes[0].set_title("EO Pre-Event"); axes[0].axis("off")

    axes[1].imshow(sar, cmap="gray")
    axes[1].set_title("SAR Post (CLAHE)"); axes[1].axis("off")

    axes[2].imshow(tgt, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth"); axes[2].axis("off")

    axes[3].imshow(pred, cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction\nthresh={threshold}"); axes[3].axis("off")

    axes[4].imshow(eo)
    axes[4].imshow(cam_pp, cmap="jet", alpha=0.5)
    axes[4].set_title(f"GradCAM++\nleak={leak_pp:.3f}"); axes[4].axis("off")

    axes[5].imshow(eo)
    axes[5].imshow(cam_lc, cmap="jet", alpha=0.5)
    axes[5].set_title(f"LayerCAM\nleak={leak_lc:.3f}"); axes[5].axis("off")

    plt.suptitle(f"GradCAM Analysis — {split.upper()}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_dir / f"gradcam_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    cam_method.remove_hooks()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--split",     default="test",
                        choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gradcam",   action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = get_device()

    # Backbone-specific output dir
    backbone  = cfg["model"]["backbone"]
    save_dir  = Path(cfg["outputs"]["results"]) / backbone
    save_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold or cfg["evaluation"]["threshold"]

    # Dataset and loader
    ds = EOSARDataset(cfg, split=args.split)
    loader = DataLoader(
        ds,
        batch_size  = cfg["evaluation"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["data"]["num_workers"],
        pin_memory  = cfg["data"]["pin_memory"],
    )

    # Load model
    # model = build_model(cfg).to(device)

    import segmentation_models_pytorch as smp
    import torch.nn as nn

    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=5,
        classes=1,
        decoder_channels=(128, 64, 32, 16, 8),
    ).to(device)
    
    model.segmentation_head = nn.Sequential(
        nn.Dropout2d(0.10),
        model.segmentation_head
    )
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.gradcam_layer = model.decoder.blocks[-1]
    print(f"Loaded weights from {args.weights}  "
          f"(trained to epoch {ckpt['epoch']}, "
          f"val_f1={ckpt['val_f1']:.4f})")

    # Evaluate
    print(f"\nEvaluating {args.split} split "
          f"(threshold={threshold})...")
    metrics = evaluate(model, loader, device, threshold)
    res     = metrics.compute()
    cm      = metrics.confusion_matrix()

    print(f"\n{args.split.upper()} RESULTS:")
    print(f"  IoU:       {res['iou']:.4f}")
    print(f"  F1:        {res['f1']:.4f}")
    print(f"  Precision: {res['precision']:.4f}")
    print(f"  Recall:    {res['recall']:.4f}")
    print(f"  TP={res['tp']}  FP={res['fp']}  "
          f"FN={res['fn']}  TN={res['tn']}")

    # Save metrics
    metrics_path = save_dir / f"metrics_{args.split}.json"
    json.dump(res, open(metrics_path, "w"), indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Plots
    plot_confusion_matrix(cm, args.split, save_dir)
    plot_qualitative(model, loader, device, threshold,
                     args.split, save_dir)

    if args.gradcam:
        print("\nRunning GradCAM analysis...")
        run_gradcam(model, loader, device, cfg,
                    threshold, args.split, save_dir)


if __name__ == "__main__":
    main()
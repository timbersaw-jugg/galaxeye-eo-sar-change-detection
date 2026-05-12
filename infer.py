"""
infer.py — Run inference on a pre/post event image pair

How inference works:
    You provide TWO files:
        --pre  : pre-event EO image  (RGB, 3 bands, .tif)
        --post : post-event SAR image (1 band, .tif)

    The model outputs a binary change mask where:
        1 = Change  (building damaged or destroyed)
        0 = No-Change

    Output files saved to --output_dir:
        change_mask.tif      : binary mask (GeoTIFF, preserves CRS)
        change_mask.png      : visual overlay on EO image
        change_probability.tif: raw probability map 0-1

Usage examples:
    # Single pair
    python infer.py \\
        --config config.yaml \\
        --weights checkpoints/best_model.pth \\
        --pre  /data/test/pre-event/scene_01_000001.tif \\
        --post /data/test/post-event/scene_01_000001.tif \\
        --threshold 0.45 \\
        --output_dir results/inference

    # Batch inference on a folder
    python infer.py \\
        --config config.yaml \\
        --weights checkpoints/best_model.pth \\
        --pre_dir  /data/test/pre-event \\
        --post_dir /data/test/post-event \\
        --output_dir results/inference_batch
"""

import argparse
import numpy as np
import torch
import yaml
import cv2
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from pathlib import Path

from models.change_detector import build_model


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
# Preprocessing — matches training pipeline exactly
# ------------------------------------------------------------------

def preprocess(pre_path, post_path, cfg):
    """
    Load and preprocess a single pre/post pair.

    Replicates training preprocessing:
        - CLAHE on SAR channel
        - Log transform on SAR channel
        - Normalize both to 0-1
        - Build 5-channel input tensor

    Returns:
        tensor:   (1, 5, H, W) ready for model
        meta:     rasterio metadata for saving output GeoTIFF
        pre_rgb:  (H, W, 3) float for visualization
        valid_mask: (H, W) bool
    """
    cl = cfg["data"]["clahe_clip_limit"]
    ts = cfg["data"]["clahe_tile_size"]
    clahe_obj = cv2.createCLAHE(clipLimit=cl, tileGridSize=(ts, ts))

    # Load pre-event EO (3ch RGB)
    with rasterio.open(pre_path) as src:
        pre  = src.read().astype(np.float32)   # (3, H, W)
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8")

    # Load post-event SAR (1ch)
    with rasterio.open(post_path) as src:
        post = src.read(1).astype(np.uint8)    # (H, W)

    H, W = pre.shape[1], pre.shape[2]

    # Valid mask
    valid = (pre.sum(axis=0) > 0) & (post > 0)

    # SAR preprocessing
    sar_clahe = clahe_obj.apply(post).astype(np.float32)
    sar_log   = np.log1p(post.astype(float))
    sar_log   = (sar_log / (sar_log.max() + 1e-8) * 255.0).astype(np.float32)

    # Normalize to 0-1
    pre_norm   = pre       / 255.0
    clahe_norm = sar_clahe / 255.0
    log_norm   = sar_log   / 255.0

    # Build 5-channel input
    inp = np.concatenate([
        pre_norm,                      # (3, H, W)
        clahe_norm[np.newaxis],        # (1, H, W)
        log_norm[np.newaxis],          # (1, H, W)
    ], axis=0)                         # (5, H, W)

    tensor = torch.from_numpy(inp).float().unsqueeze(0)  # (1, 5, H, W)

    # EO RGB for visualization
    pre_rgb = np.clip(pre_norm.transpose(1, 2, 0), 0, 1)  # (H, W, 3)

    return tensor, meta, pre_rgb, valid, H, W


# ------------------------------------------------------------------
# Tiled inference for large images
# ------------------------------------------------------------------

def tiled_inference(model, tensor, device,
                    tile_size=512, overlap=64, threshold=0.45):
    """
    Run inference on large images using overlapping tiles.
    Averages predictions in overlap regions.

    tile_size: size of each tile (must match training crop size)
    overlap:   overlap between adjacent tiles in pixels

    This handles 1024x1024 images efficiently without OOM.
    """
    _, C, H, W = tensor.shape
    stride     = tile_size - overlap
    prob_map   = np.zeros((H, W), dtype=np.float32)
    count_map  = np.zeros((H, W), dtype=np.float32)

    model.eval()

    ys = list(range(0, max(1, H - tile_size + 1), stride))
    xs = list(range(0, max(1, W - tile_size + 1), stride))

    # Ensure last tile covers edge
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)

                tile = tensor[:, :, y1:y2, x1:x2].to(device)
                logit= model(tile)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()

                prob_map[y1:y2, x1:x2] += prob
                count_map[y1:y2, x1:x2]+= 1.0

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1.0)
    prob_map  = prob_map / count_map

    mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, mask


# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------

def save_outputs(prob_map, mask, pre_rgb, valid_mask,
                 meta, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Binary change mask as GeoTIFF
    mask_path = output_dir / f"{stem}_change_mask.tif"
    with rasterio.open(mask_path, "w", **meta) as dst:
        dst.write(mask[np.newaxis].astype(np.uint8))
    print(f"Saved: {mask_path}")

    # 2. Probability map as GeoTIFF (float32)
    prob_meta = meta.copy()
    prob_meta.update(dtype="float32")
    prob_path = output_dir / f"{stem}_change_prob.tif"
    with rasterio.open(prob_path, "w", **prob_meta) as dst:
        dst.write(prob_map[np.newaxis])
    print(f"Saved: {prob_path}")

    # 3. Visual overlay PNG
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(pre_rgb)
    axes[0].set_title("EO Pre-Event (RGB)")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Change Mask\n(1=Changed, 0=No-Change)")
    axes[1].axis("off")

    # Overlay: EO + change mask
    overlay = pre_rgb.copy()
    change_pixels = mask.astype(bool)
    overlay[change_pixels, 0] = 1.0   # Red channel
    overlay[change_pixels, 1] *= 0.3  # Reduce green
    overlay[change_pixels, 2] *= 0.3  # Reduce blue

    # Nodata regions in dark gray
    nodata = ~valid_mask
    overlay[nodata] = 0.15

    axes[2].imshow(overlay)
    axes[2].set_title("Change Overlay\n(Red=Changed)")
    axes[2].axis("off")

    plt.suptitle(f"Inference Result: {stem}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    png_path = output_dir / f"{stem}_overlay.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")

    # Print stats
    valid_pixels  = valid_mask.sum()
    change_pixels = (mask.astype(bool) & valid_mask).sum()
    change_pct    = 100 * change_pixels / (valid_pixels + 1e-8)
    print(f"\nStats for {stem}:")
    print(f"  Valid pixels:  {valid_pixels:,}")
    print(f"  Change pixels: {change_pixels:,} ({change_pct:.2f}%)")


# ------------------------------------------------------------------
# Single pair inference
# ------------------------------------------------------------------

def infer_single(model, pre_path, post_path,
                 cfg, device, threshold, output_dir):
    print(f"\nProcessing: {Path(pre_path).name}")

    tensor, meta, pre_rgb, valid_mask, H, W = preprocess(
        pre_path, post_path, cfg
    )

    tile_size = cfg["data"]["crop_size"]
    prob_map, mask = tiled_inference(
        model, tensor, device,
        tile_size=tile_size, overlap=64, threshold=threshold
    )

    stem = Path(pre_path).stem
    save_outputs(prob_map, mask, pre_rgb, valid_mask,
                 meta, output_dir, stem)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EO-SAR Change Detection Inference"
    )
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--weights",    required=True,
                        help="Path to model checkpoint .pth")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Sigmoid threshold (default: from config)")
    parser.add_argument("--output_dir", default="results/inference")

    # Single pair
    parser.add_argument("--pre",  default=None,
                        help="Path to pre-event EO .tif file")
    parser.add_argument("--post", default=None,
                        help="Path to post-event SAR .tif file")

    # Batch mode
    parser.add_argument("--pre_dir",  default=None,
                        help="Directory of pre-event .tif files")
    parser.add_argument("--post_dir", default=None,
                        help="Directory of post-event .tif files")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = get_device()
    threshold = args.threshold or cfg["evaluation"]["threshold"]

    print(f"Device:    {device}")
    print(f"Weights:   {args.weights}")
    print(f"Threshold: {threshold}")

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded (epoch={ckpt['epoch']}, "
          f"val_f1={ckpt['val_f1']:.4f})")

    # Infer
    if args.pre and args.post:
        # Single pair mode
        infer_single(model, args.pre, args.post,
                     cfg, device, threshold, args.output_dir)

    elif args.pre_dir and args.post_dir:
        # Batch mode
        pre_files  = sorted(Path(args.pre_dir).glob("*.tif"))
        post_files = sorted(Path(args.post_dir).glob("*.tif"))

        assert len(pre_files) == len(post_files), \
            "Mismatch: pre and post directories have different file counts"

        print(f"\nBatch mode: {len(pre_files)} pairs")
        for pre_f, post_f in zip(pre_files, post_files):
            assert pre_f.name == post_f.name, \
                f"Name mismatch: {pre_f.name} vs {post_f.name}"
            infer_single(model, pre_f, post_f,
                         cfg, device, threshold, args.output_dir)

        print(f"\n✅ Batch complete. Results in: {args.output_dir}")

    else:
        parser.error(
            "Provide either --pre + --post  OR  --pre_dir + --post_dir"
        )


if __name__ == "__main__":
    main()
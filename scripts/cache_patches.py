# scripts/cache_patches.py
# Run ONCE before training
# Extracts all valid 256x256 patches from tif files
# Saves as compressed .npz
# Training then reads .npz instead of .tif

import numpy as np
import rasterio
import cv2
from pathlib import Path
from tqdm import tqdm

REMAP = {0: 0, 1: 0, 2: 1, 3: 1}

def cache_split(data_root, split, patch_size=256,
                stride=128, min_valid=0.4, min_change=0):
    """
    Extract and save all valid patches from a split.
    stride < patch_size means overlapping patches.
    Saves each patch as individual .npz file.
    """
    root      = Path(data_root)
    out_dir   = root / f"cache_{patch_size}_s{stride}" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_files = sorted((root/split/"pre-event").glob("*.tif"))
    post_files= sorted((root/split/"post-event").glob("*.tif"))
    tgt_files = sorted((root/split/"target").glob("*.tif"))

    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    total_patches  = 0
    change_patches = 0
    ps             = patch_size

    print(f"\nCaching {split} split → {out_dir}")

    for pre_f, post_f, tgt_f in tqdm(
        zip(pre_files, post_files, tgt_files),
        total=len(pre_files)
    ):
        with rasterio.open(pre_f) as src:
            pre = src.read().astype(np.float32)     # (3,H,W)
        with rasterio.open(post_f) as src:
            post = src.read(1).astype(np.uint8)     # (H,W)
        with rasterio.open(tgt_f) as src:
            tgt = src.read(1)

        # Remap target
        tgt_bin = np.zeros_like(tgt, dtype=np.uint8)
        tgt_bin[(tgt == 2) | (tgt == 3)] = 1

        # SAR preprocessing
        sar_clahe = clahe_obj.apply(post).astype(np.float32)
        sar_log   = np.log1p(post.astype(float))
        sar_log   = (sar_log/(sar_log.max()+1e-8)*255).astype(np.float32)

        # Valid mask
        valid = (pre.sum(0) > 0) & (post > 0)

        _, H, W = pre.shape

        for y in range(0, H - ps + 1, stride):
            for x in range(0, W - ps + 1, stride):
                # Crop
                pre_c   = pre[:, y:y+ps, x:x+ps]
                cla_c   = sar_clahe[y:y+ps, x:x+ps]
                log_c   = sar_log  [y:y+ps, x:x+ps]
                tgt_c   = tgt_bin  [y:y+ps, x:x+ps]
                vm_c    = valid    [y:y+ps, x:x+ps]

                # Skip if not enough valid pixels
                if vm_c.mean() < min_valid:
                    continue

                has_change = tgt_c.sum() > min_change

                # Build 5ch input (store as uint8 to save disk)
                inp = np.stack([
                    pre_c[0].astype(np.uint8),
                    pre_c[1].astype(np.uint8),
                    pre_c[2].astype(np.uint8),
                    cla_c.astype(np.uint8),
                    log_c.astype(np.uint8),
                ], axis=0)  # (5, ps, ps)

                stem      = pre_f.stem
                patch_id  = f"{stem}_y{y}_x{x}"
                tag       = "change" if has_change else "nochange"
                fname     = out_dir / f"{tag}_{patch_id}.npz"

                np.savez(
                    fname,
                    input      = inp,
                    target     = tgt_c,
                    valid_mask = vm_c.astype(np.uint8),
                )

                total_patches  += 1
                change_patches += int(has_change)

    print(f"  Total patches:  {total_patches:,}")
    print(f"  Change patches: {change_patches:,} "
          f"({100*change_patches/max(total_patches,1):.1f}%)")
    print(f"  Saved to: {out_dir}")
    return total_patches, change_patches


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--patch_size",  type=int, default=256)
    parser.add_argument("--stride",      type=int, default=128)
    parser.add_argument("--min_valid",   type=float, default=0.4)
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        cache_split(
            args.data_root, split,
            patch_size = args.patch_size,
            stride     = args.stride,
            min_valid  = args.min_valid,
        )

    print("\n✅ Caching complete. Update config.yaml:")
    print(f"   data.use_cache: true")
    print(f"   data.cache_dir: your_data_root/cache_{args.patch_size}_s{args.stride}")
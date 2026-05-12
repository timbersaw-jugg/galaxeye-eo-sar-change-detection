# diagnose_test.py
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_ROOT  = Path(cfg["data"]["root"])
CACHE_DIR  = Path(cfg["data"]["cache_dir"])

# ── Check 1: What scenes are in test vs val? ──────────────
print("TEST scene names:")
test_files = sorted((DATA_ROOT/"test"/"pre-event").glob("*.tif"))
val_files  = sorted((DATA_ROOT/"val"/"pre-event").glob("*.tif"))

test_scenes = set(f.name.split("_")[1] for f in test_files)
val_scenes  = set(f.name.split("_")[1] for f in val_files)

print(f"  Test scene IDs: {sorted(test_scenes)}")
print(f"  Val  scene IDs: {sorted(val_scenes)}")
print(f"  Overlap: {test_scenes & val_scenes}")
print(f"  Test only: {test_scenes - val_scenes}")

# ── Check 2: Class distribution in test patches ───────────
from data.dataset import CachedEOSARDataset
from torch.utils.data import DataLoader
import torch

test_ds = CachedEOSARDataset(
    cfg["data"]["cache_dir"],
    split="test", augment=False, oversample_change=1
)
print(f"\nTest patches: {len(test_ds)}")

change_total = 0
valid_total  = 0
for i in range(min(100, len(test_ds))):
    d = test_ds[i]
    change_total += (d["target"] == 1).sum().item()
    valid_total  += d["valid_mask"].sum().item()

print(f"Change pixel ratio in test: "
      f"{change_total/max(valid_total,1):.4f}")

# ── Check 3: Visual comparison val vs test ────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Val samples
val_ds = CachedEOSARDataset(
    cfg["data"]["cache_dir"],
    split="val", augment=False, oversample_change=1
)

for i, (ds, row, title) in enumerate([
    (val_ds,  0, "VAL"),
    (test_ds, 1, "TEST")
]):
    for j in range(4):
        idx  = j * (len(ds) // 4)
        d    = ds[idx]
        inp  = d["input"]
        tgt  = d["target"].numpy()

        eo = inp[:3].permute(1,2,0).numpy()
        eo = np.clip(eo, 0, 1)

        axes[i][j].imshow(eo)
        change_pct = 100 * tgt.mean()
        axes[i][j].set_title(
            f"{title} sample {j}\n"
            f"change={change_pct:.1f}%"
        )
        axes[i][j].axis("off")

plt.suptitle("Val vs Test Visual Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("val_vs_test_comparison.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("\nSaved: val_vs_test_comparison.png")

# ── Check 4: Image statistics comparison ──────────────────
print("\nImage statistics comparison:")
for split in ["val", "test"]:
    files = sorted(
        (DATA_ROOT/split/"pre-event").glob("*.tif")
    )[:10]
    means, stds = [], []
    for f in files:
        with rasterio.open(f) as src:
            d = src.read().astype(float)
            valid = d.sum(0) > 0
            if valid.sum() > 0:
                means.append(d[:, valid].mean())
                stds.append(d[:, valid].std())
    print(f"  {split}: EO mean={np.mean(means):.2f} "
          f"std={np.mean(stds):.2f}")

# SAR statistics
for split in ["val", "test"]:
    files = sorted(
        (DATA_ROOT/split/"post-event").glob("*.tif")
    )[:10]
    means, stds = [], []
    for f in files:
        with rasterio.open(f) as src:
            d = src.read(1).astype(float)
            valid = d > 0
            if valid.sum() > 0:
                means.append(d[valid].mean())
                stds.append(d[valid].std())
    print(f"  {split}: SAR mean={np.mean(means):.2f} "
          f"std={np.mean(stds):.2f}")
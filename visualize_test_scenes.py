# visualize_test_scenes.py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = Path(cfg["data"]["root"])

pre_files  = sorted((DATA_ROOT/"test"/"pre-event").glob("*.tif"))
post_files = sorted((DATA_ROOT/"test"/"post-event").glob("*.tif"))
tgt_files  = sorted((DATA_ROOT/"test"/"target").glob("*.tif"))

print(f"Test files: {len(pre_files)}")
print("File names:")
for f in pre_files[:5]:
    print(f"  {f.name}")

# Visualize first 3 test pairs
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for i in range(min(3, len(pre_files))):
    with rasterio.open(pre_files[i]) as src:
        pre = src.read()
        print(f"\nTest file {i}: {pre_files[i].name}")
        print(f"  Pre  shape: {src.count}ch "
              f"{src.height}x{src.width} "
              f"dtype={src.dtypes[0]}")
        print(f"  Pre  range: {pre.min()}-{pre.max()}")

    with rasterio.open(post_files[i]) as src:
        post = src.read(1)
        print(f"  Post shape: 1ch "
              f"{src.height}x{src.width} "
              f"dtype={src.dtypes[0]}")
        print(f"  Post range: {post.min()}-{post.max()}")

    with rasterio.open(tgt_files[i]) as src:
        tgt = src.read(1)
    tgt_bin = ((tgt == 2) | (tgt == 3)).astype(np.uint8)

    def norm(x):
        mn = np.percentile(x[x>0], 2) if (x>0).any() else 0
        mx = np.percentile(x[x>0], 98) if (x>0).any() else 1
        return np.clip((x.astype(float)-mn)/(mx-mn+1e-8), 0, 1)

    eo = np.stack([norm(pre[0]), norm(pre[1]),
                   norm(pre[2])], axis=-1)

    axes[i][0].imshow(eo)
    axes[i][0].set_title(f"Test EO [{i}]\n"
                          f"{pre_files[i].name[:30]}")
    axes[i][0].axis("off")

    axes[i][1].imshow(post, cmap="gray")
    axes[i][1].set_title(f"Test SAR [{i}]")
    axes[i][1].axis("off")

    axes[i][2].imshow(tgt_bin, cmap="Reds")
    axes[i][2].set_title(f"GT change={tgt_bin.mean():.3f}")
    axes[i][2].axis("off")

    # Show raw model output distribution
    import torch, cv2
    from models.change_detector import build_model

    model = build_model(cfg)
    ckpt  = torch.load(
        "checkpoints/resnet18/best_model.pth",
        map_location="cpu"
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    clahe = cv2.createCLAHE(clipLimit=2.0,
                             tileGridSize=(8,8))
    sar_c = clahe.apply(post.astype(np.uint8))
    sar_l = np.log1p(post.astype(float))
    sar_l = (sar_l/(sar_l.max()+1e-8)*255).astype(np.uint8)

    inp = np.stack([
        pre[0], pre[1], pre[2],
        sar_c, sar_l
    ], axis=0).astype(np.float32) / 255.0

    # Use center crop for visualization
    H, W = inp.shape[1], inp.shape[2]
    cs   = 256
    y    = max(0, (H-cs)//2)
    x    = max(0, (W-cs)//2)
    inp_crop = inp[:, y:y+cs, x:x+cs]

    with torch.no_grad():
        t = torch.from_numpy(inp_crop).unsqueeze(0)
        out = torch.sigmoid(model(t)).squeeze().numpy()

    axes[i][3].imshow(out, cmap="hot", vmin=0, vmax=1)
    axes[i][3].set_title(f"Model prob\n"
                          f"max={out.max():.3f} "
                          f"mean={out.mean():.4f}")
    axes[i][3].axis("off")

plt.suptitle("Test Scenes 09-10 — Model Output Analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("test_scene_analysis.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("\nSaved: test_scene_analysis.png")
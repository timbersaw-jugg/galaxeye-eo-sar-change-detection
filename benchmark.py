# benchmark.py — isolates each bottleneck
import time
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import CachedEOSARDataset
from models.change_detector import build_model

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Test 1: Raw dataloader speed ──────────────────────────
print("TEST 1: Dataloader speed (no model)")
ds = CachedEOSARDataset(
    cfg["data"]["cache_dir"],
    split="train",
    augment=False,
    oversample_change=1
)
loader = DataLoader(ds, batch_size=8,
                    shuffle=True, num_workers=0)

t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 20:
        break
elapsed = (time.time() - t0) / 20
print(f"  Per batch (data only): {elapsed:.3f}s")
print(f"  Estimated epoch (data only): "
      f"{elapsed * len(loader) / 60:.1f} min")

# ── Test 2: Forward pass speed ────────────────────────────
print("\nTEST 2: Forward pass speed (no data loading)")
model = build_model(cfg)
model.eval()
dummy = torch.randn(8, 5, 256, 256)

# Warmup
with torch.no_grad():
    _ = model(dummy)

t0 = time.time()
for _ in range(20):
    with torch.no_grad():
        out = model(dummy)
elapsed = (time.time() - t0) / 20
print(f"  Per batch (forward only): {elapsed:.3f}s")

# ── Test 3: Forward + backward ────────────────────────────
print("\nTEST 3: Forward + backward (full train step)")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

t0 = time.time()
for _ in range(10):
    optimizer.zero_grad()
    out  = model(dummy)
    loss = out.mean()
    loss.backward()
    optimizer.step()
elapsed = (time.time() - t0) / 10
print(f"  Per batch (fwd+bwd): {elapsed:.3f}s")
print(f"  Estimated epoch (fwd+bwd only): "
      f"{elapsed * len(loader) / 60:.1f} min")

# Add to benchmark.py — run after adding MobileNetV2

print("\nTEST 4: MobileNetV2 forward+backward")
cfg["model"]["backbone"] = "mobilenet_v2"
model_mv2 = build_model(cfg)
model_mv2.train()
optimizer2 = torch.optim.AdamW(
    model_mv2.parameters(), lr=1e-4
)

# Warmup
out = model_mv2(dummy)
out.mean().backward()
optimizer2.zero_grad()

t0 = time.time()
for _ in range(10):
    optimizer2.zero_grad()
    out  = model_mv2(dummy)
    loss = out.mean()
    loss.backward()
    optimizer2.step()
elapsed = (time.time() - t0) / 10

n_batches = 15713
print(f"  Per batch (fwd+bwd): {elapsed:.3f}s")
print(f"  Estimated epoch:     "
      f"{elapsed * n_batches / 60:.1f} min")
print(f"  Speedup vs ResNet18: "
      f"{1.953/elapsed:.1f}x faster")

# ── Summary ───────────────────────────────────────────────
print("\nSUMMARY")
print(f"  Batches per epoch: {len(loader)}")
print(f"  If bottleneck is data:  check num_workers, cache")
print(f"  If bottleneck is model: need lighter architecture")
print(f"  If both slow: need GPU or lighter everything")
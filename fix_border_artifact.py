# fix_border_artifact.py
# Erode the valid mask to remove border pixels
# Then re-evaluate test set

import torch
import numpy as np
import yaml
import cv2
from torch.utils.data import DataLoader
from data.dataset import CachedEOSARDataset
from models.change_detector import build_model
from utils.metrics import ChangeMetrics

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu")
model  = build_model(cfg).to(device)
ckpt   = torch.load(
    "checkpoints/resnet18/best_model.pth",
    map_location=device
)
model.load_state_dict(ckpt["model_state"])
model.eval()

test_ds = CachedEOSARDataset(
    cfg["data"]["cache_dir"],
    split="test", augment=False, oversample_change=1
)
loader = DataLoader(
    test_ds, batch_size=1,
    shuffle=False, num_workers=0
)

print("Evaluating TEST with border erosion...")
print(f"{'Erosion':>10} {'F1':>8} {'IoU':>8} "
      f"{'Prec':>8} {'Recall':>8}")
print("-" * 50)

# Try different erosion kernel sizes
for erosion_px in [0, 5, 10, 15, 20, 30]:
    metrics = ChangeMetrics(threshold=0.45)

    kernel = np.ones(
        (erosion_px*2+1, erosion_px*2+1),
        np.uint8
    ) if erosion_px > 0 else None

    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            vm  = batch["valid_mask"]

            # Erode valid mask to remove border pixels
            if erosion_px > 0:
                vm_np = vm[0].numpy().astype(np.uint8)
                vm_eroded = cv2.erode(vm_np, kernel)
                vm_use = torch.from_numpy(
                    vm_eroded.astype(bool)
                ).unsqueeze(0).to(device)
            else:
                vm_use = vm.to(device)

            out = model(inp)
            metrics.update(out, tgt, vm_use)

    res = metrics.compute()
    print(f"{erosion_px:>10}px  "
          f"{res['f1']:>8.4f}  "
          f"{res['iou']:>8.4f}  "
          f"{res['precision']:>8.4f}  "
          f"{res['recall']:>8.4f}")
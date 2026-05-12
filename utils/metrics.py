import numpy as np
import torch


class ChangeMetrics:
    """
    Computes IoU, Precision, Recall, F1 for change class (label=1).
    Accumulates over batches then computes final metrics.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits, targets, valid_mask=None):
        """
        logits:     (B, 1, H, W) raw logits
        targets:    (B, H, W)    long {0, 1}
        valid_mask: (B, H, W)    bool
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= self.threshold).long()

            if valid_mask is not None:
                preds   = preds[valid_mask]
                targets = targets[valid_mask]

            self.tp += int(((preds == 1) & (targets == 1)).sum())
            self.fp += int(((preds == 1) & (targets == 0)).sum())
            self.fn += int(((preds == 0) & (targets == 1)).sum())
            self.tn += int(((preds == 0) & (targets == 0)).sum())

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps       = 1e-8
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = tp / (tp + fp + fn + eps)
        return {
            "iou":       round(iou,       4),
            "f1":        round(f1,        4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def confusion_matrix(self):
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp],
        ])

    def print_results(self, split=""):
        res = self.compute()
        tag = f"[{split}] " if split else ""
        print(f"{tag}IoU={res['iou']:.4f}  F1={res['f1']:.4f}  "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}  "
              f"TP={res['tp']}  FP={res['fp']}  "
              f"FN={res['fn']}  TN={res['tn']}")
        return res


def tune_threshold(model, loader, device, thresholds=None):
    """
    Find optimal sigmoid threshold on validation set by F1.
    Returns best threshold and results table.
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.70]

    model.eval()
    results = []

    for t in thresholds:
        metrics = ChangeMetrics(threshold=t)
        with torch.no_grad():
            for batch in loader:
                inp = batch["input"].to(device)
                tgt = batch["target"].to(device)
                vm  = batch["valid_mask"].to(device)
                out = model(inp)
                metrics.update(out, tgt, vm)
        res = metrics.compute()
        results.append((t, res["f1"], res["iou"],
                        res["precision"], res["recall"]))
        print(f"  thresh={t:.2f}  F1={res['f1']:.4f}  "
              f"IoU={res['iou']:.4f}  "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n✅ Best threshold: {best[0]}  (F1={best[1]:.4f})")
    return best[0], results
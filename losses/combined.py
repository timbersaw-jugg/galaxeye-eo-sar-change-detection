import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Directly optimizes overlap between prediction and target.
    Handles class imbalance by focusing on region overlap.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask=None):
        probs = torch.sigmoid(logits).squeeze(1)   # (B, H, W)
        tgt   = targets.float()

        if valid_mask is not None:
            vm    = valid_mask.float()
            probs = probs * vm
            tgt   = tgt   * vm

        inter = (probs * tgt).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        dice  = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss — reduces weight on easy negatives.
    Forces model to focus on hard, misclassified change pixels.
    alpha: weight for positive class
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, valid_mask=None):
        tgt    = targets.float().unsqueeze(1)       # (B, 1, H, W)
        bce    = F.binary_cross_entropy_with_logits(
            logits, tgt, reduction="none"
        )
        probs  = torch.sigmoid(logits)
        p_t    = probs * tgt + (1 - probs) * (1 - tgt)
        weight = self.alpha * (1 - p_t) ** self.gamma
        loss   = weight * bce

        if valid_mask is not None:
            vm   = valid_mask.float().unsqueeze(1)
            loss = loss * vm
            return loss.sum() / (vm.sum() + 1e-8)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for EO-SAR change detection.

    Formula:
        L = w_bce * BCE(pos_weight) + w_dice * Dice + w_focal * Focal

    Design rationale:
        BCE  : pixel-level supervision with pos_weight for imbalance
        Dice : region-level overlap, robust to class imbalance
        Focal: focuses on hard change pixels, reduces easy negative dominance

    pos_weight = ~45 (inverse of change pixel ratio in validation set)
    """
    def __init__(self, cfg):
        super().__init__()
        self.w_bce   = cfg["loss"]["bce_weight"]
        self.w_dice  = cfg["loss"]["dice_weight"]
        self.w_focal = cfg["loss"]["focal_weight"]

        self.dice  = DiceLoss(smooth=1.0)
        self.focal = FocalLoss(
            alpha = cfg["loss"]["focal_alpha"],
            gamma = cfg["loss"]["focal_gamma"],
        )

        pw = torch.tensor([cfg["loss"]["pos_weight"]])
        self.register_buffer("pos_weight", pw)

    def forward(self, logits, targets, valid_mask=None):
        """
        logits:     (B, 1, H, W) raw model output
        targets:    (B, H, W)    long {0, 1}
        valid_mask: (B, H, W)    bool, True where data is valid
        """
        tgt = targets.float().unsqueeze(1)          # (B, 1, H, W)

        # BCE with pos_weight — masked to valid region
        if valid_mask is not None:
            vm      = valid_mask.float().unsqueeze(1)
            bce_raw = F.binary_cross_entropy_with_logits(
                logits, tgt,
                pos_weight = self.pos_weight.to(logits.device),
                reduction  = "none",
            )
            bce_loss = (bce_raw * vm).sum() / (vm.sum() + 1e-8)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, tgt,
                pos_weight = self.pos_weight.to(logits.device),
            )

        dice_loss  = self.dice(logits, targets, valid_mask)
        focal_loss = self.focal(logits, targets, valid_mask)

        total = (self.w_bce  * bce_loss +
                 self.w_dice * dice_loss +
                 self.w_focal* focal_loss)

        return total, {
            "bce":   bce_loss.item(),
            "dice":  dice_loss.item(),
            "focal": focal_loss.item(),
            "total": total.item(),
        }
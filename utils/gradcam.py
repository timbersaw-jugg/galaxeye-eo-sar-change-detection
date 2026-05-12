import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# GradCAM++ — Primary
# Best for small object localization (buildings)
# Uses weighted spatial gradients
# ------------------------------------------------------------------

class GradCAMPlusPlus:
    """
    GradCAM++ for CNN backbones (ResNet18, ConvNeXt).

    Math:
        alpha_k_ij = (∂²y / ∂A^k_ij²) /
                     (2 * ∂²y/∂A^k_ij² + Σ_ab A^k_ab * ∂³y/∂A^k_ij³)

        L = ReLU(Σ_k (Σ_ij alpha_k_ij * ReLU(∂y/∂A^k_ij)) * A^k)

    Advantage over GradCAM:
        Weights each spatial location differently
        Better localization of small/multiple objects

    Hook type: register_full_backward_hook (not deprecated)
    Target:    model.dec1.conv[1]  (last decoder ConvBNReLU)

    CRITICAL:
        Do NOT detach in hook callbacks
        Detach only when computing final CAM array
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._handles    = []

        h1 = target_layer.register_forward_hook(self._save_activation)
        h2 = target_layer.register_full_backward_hook(self._save_gradient)
        self._handles = [h1, h2]

    def _save_activation(self, module, input, output):
        # NO detach — keep in computation graph
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # NO detach — keep in computation graph
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        """
        input_tensor: (1, C, H, W) single image, on correct device
        Returns: (H, W) numpy array, values 0-1
        """
        self.model.eval()
        inp = input_tensor.clone().requires_grad_(True)

        # Forward
        logits = self.model(inp)                    # (1, 1, H, W)

        # Backward on mean logit (change detection score)
        self.model.zero_grad()
        logits.mean().backward()

        # Detach NOW — after backward
        grads = self.gradients.detach()             # (1, C, h, w)
        acts  = self.activations.detach()           # (1, C, h, w)

        # GradCAM++ weighting
        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        acts_sum  = acts.sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)

        alpha_num = grads_sq
        alpha_den = 2.0 * grads_sq + acts_sum * grads_cu + 1e-8
        alpha     = alpha_num / alpha_den                # (1,C,h,w)

        weights   = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam       = F.relu((weights * acts).sum(dim=1, keepdim=True))

        # Normalize and upsample
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=inp.shape[2:],
                            mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy()          # (H, W)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


# ------------------------------------------------------------------
# LayerCAM — Secondary
# Better boundary detail for report figures
# Per-pixel spatial gradient weighting
# ------------------------------------------------------------------

class LayerCAM:
    """
    LayerCAM for CNN backbones.

    Math:
        w_k_ij = ReLU(∂y / ∂A^k_ij)   per spatial location
        L = ReLU(Σ_k w_k_ij * A^k_ij)

    Advantage over GradCAM++:
        Preserves finer spatial detail
        Better building boundary delineation
        More precise for dense prediction tasks

    Use case: report visualizations, boundary analysis
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._handles    = []

        h1 = target_layer.register_forward_hook(self._save_activation)
        h2 = target_layer.register_full_backward_hook(self._save_gradient)
        self._handles = [h1, h2]

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()
        inp = input_tensor.clone().requires_grad_(True)

        logits = self.model(inp)
        self.model.zero_grad()
        logits.mean().backward()

        grads = self.gradients.detach()
        acts  = self.activations.detach()

        # Per-pixel spatial weighting
        weights = F.relu(grads)                     # (1, C, h, w)
        cam     = F.relu((weights * acts).sum(dim=1, keepdim=True))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=inp.shape[2:],
                            mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


# ------------------------------------------------------------------
# EigenCAM — Transformer fallback (stub)
# Gradient-free, works on Swin/SegFormer feature maps
# Activate when backbone switches to transformer
# ------------------------------------------------------------------

class EigenCAM:
    """
    EigenCAM for transformer backbones (Swin, SegFormer).

    Math:
        Flatten activations A: (C, h*w)
        SVD: A = U Σ V^T
        CAM = first principal component V[:, 0] reshaped to (h, w)

    Use when:
        backbone = swin_transformer or segformer
        GradCAM/GradCAM++ do NOT work on attention layers directly

    TODO: activate when backbone switches to transformer
    Current backbone: resnet18 → use GradCAMPlusPlus instead
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self._handles    = []

        h1 = target_layer.register_forward_hook(self._save_activation)
        self._handles = [h1]
        # Note: NO backward hook needed — gradient-free method

    def _save_activation(self, module, input, output):
        self.activations = output.detach()          # safe to detach — no backward

    def generate(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        acts = self.activations                     # (1, C, h, w)
        b, c, h, w = acts.shape

        # Flatten spatial dims
        A = acts.squeeze(0).view(c, h * w).cpu().numpy()  # (C, h*w)

        # SVD — first principal component
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        cam = Vt[0].reshape(h, w)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample
        cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
        cam_t = F.interpolate(cam_t, size=input_tensor.shape[2:],
                              mode="bilinear", align_corners=False)
        return cam_t.squeeze().numpy()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


# ------------------------------------------------------------------
# Safety check
# ------------------------------------------------------------------

def boundary_leak_check(cam, valid_mask, threshold=0.5):
    """
    Check if GradCAM activations leak into nodata regions.

    cam:        (H, W) numpy float 0-1
    valid_mask: (H, W) bool, True = valid data region
    threshold:  activation threshold to consider 'high'

    Returns leak_ratio — should be < 0.05 (5%)
    A ratio > 0.05 suggests model is using nodata artifacts.
    """
    high_act   = cam > threshold
    leak       = high_act & (~valid_mask)
    leak_ratio = leak.sum() / (high_act.sum() + 1e-8)
    return float(leak_ratio)


def build_gradcam(model, cfg):
    """
    Build the appropriate CAM method based on config and backbone.
    """
    backbone = cfg["model"]["backbone"]
    method   = cfg["gradcam"]["method"]

    target_layer = model.gradcam_layer

    if backbone in ("resnet18", "convnext_tiny"):
        if method == "gradcam_pp":
            print(f"GradCAM: GradCAMPlusPlus on {backbone}")
            return GradCAMPlusPlus(model, target_layer)
        elif method == "layercam":
            print(f"GradCAM: LayerCAM on {backbone}")
            return LayerCAM(model, target_layer)
        else:
            print(f"GradCAM: GradCAMPlusPlus (default) on {backbone}")
            return GradCAMPlusPlus(model, target_layer)
    else:
        # Transformer backbone — use EigenCAM
        print(f"GradCAM: EigenCAM (transformer mode) on {backbone}")
        return EigenCAM(model, target_layer)
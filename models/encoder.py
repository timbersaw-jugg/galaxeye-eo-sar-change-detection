"""
models/encoder.py — Swappable Backbone Encoders

Supported backbones:
    resnet18      : 14.5M params, fast, good baseline
    convnext_tiny : 28M params,  modern CNN, better features

All encoders:
    - Accept 64ch input from Stem (not raw RGB)
    - Output 5 feature maps at different scales
    - Return dict with keys: e1, e2, e3, e4, e5, channels

To add a new backbone:
    1. Add a new class following the same interface
    2. Register it in ENCODER_REGISTRY at bottom of file
    3. Update config.yaml: model.backbone = your_new_backbone

GradCAM compatibility:
    CNN backbones (resnet18, convnext_tiny):
        Use GradCAMPlusPlus or LayerCAM
        Hook: register_full_backward_hook on dec1.conv[1]

    Transformer backbones (future — swin, segformer):
        Use EigenCAM or AttentionRollout
        Hook: register_forward_hook on last stage output
        See utils/gradcam.py EigenCAM class
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


# ------------------------------------------------------------------
# ResNet18 Encoder
# ------------------------------------------------------------------

class ResNet18Encoder(nn.Module):
    """
    ResNet18 backbone encoder.

    Input:  (B, 64, H, W)    from Stem
    Outputs:
        e1: (B, 64,  H/2,  W/2)
        e2: (B, 64,  H/4,  W/4)
        e3: (B, 128, H/8,  W/8)
        e4: (B, 256, H/16, W/16)
        e5: (B, 512, H/32, W/32)

    Receptive field at e5: ~407px (covers building clusters)
    Pretrained: ImageNet weights via torchvision

    Architectural note:
        conv1 is patched from 3ch→64ch to accept Stem output.
        Original pretrained conv1 weights are NOT loaded for this layer
        since channel count changes — all other layers keep ImageNet init.
    """
    channels = [64, 64, 128, 256, 512]

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        base    = tvm.resnet18(weights=weights)

        # Patch conv1: 3ch → 64ch (stem output)
        base.conv1 = nn.Conv2d(64, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc2 = base.layer1
        self.enc3 = base.layer2
        self.enc4 = base.layer3
        self.enc5 = base.layer4

    def forward(self, x):
        e1  = self.enc1(x)      # (B, 64,  H/2,  W/2)
        e1p = self.pool(e1)     # (B, 64,  H/4,  W/4)
        e2  = self.enc2(e1p)    # (B, 64,  H/4,  W/4)
        e3  = self.enc3(e2)     # (B, 128, H/8,  W/8)
        e4  = self.enc4(e3)     # (B, 256, H/16, W/16)
        e5  = self.enc5(e4)     # (B, 512, H/32, W/32)
        return {
            "e1": e1, "e2": e2, "e3": e3,
            "e4": e4, "e5": e5,
            "skip1": e1,        # skip for decoder
        }


# ------------------------------------------------------------------
# ConvNeXt V2 Tiny Encoder
# ------------------------------------------------------------------

class ConvNeXtTinyEncoder(nn.Module):
    """
    ConvNeXt-Tiny backbone encoder.

    Input:  (B, 64, H, W)    from Stem
    Outputs:
        e1: (B, 96,  H/4,  W/4)
        e2: (B, 96,  H/4,  W/4)
        e3: (B, 192, H/8,  W/8)
        e4: (B, 384, H/16, W/16)
        e5: (B, 768, H/32, W/32)

    Pretrained: ImageNet-1K weights via torchvision

    Architectural note:
        features[0][0] (stem conv) is patched 3ch → 64ch input,
        kernel_size=4 stride=4 preserved from original design.
        MUST patch BEFORE assigning enc layers — order is critical.

    GradCAM note:
        Same CNN backbone → GradCAMPlusPlus works fine
        Target: model.gradcam_layer (dec1.conv[1])
    """
    channels = [64, 96, 192, 384, 768]

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base    = tvm.convnext_tiny(weights=weights)

        # Patch FIRST then assign — critical order
        base.features[0][0] = nn.Conv2d(64, 96,
                                         kernel_size=4, stride=4)

        self.enc1 = nn.Sequential(base.features[0])
        self.enc2 = nn.Sequential(base.features[1], base.features[2])
        self.enc3 = nn.Sequential(base.features[3], base.features[4])
        self.enc4 = nn.Sequential(base.features[5], base.features[6])
        self.enc5 = nn.Sequential(base.features[7])

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        return {
            "e1": e1, "e2": e2, "e3": e3,
            "e4": e4, "e5": e5,
            "skip1": e1,
        }
# Add to models/encoder.py

class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 backbone encoder.
    Uses depthwise separable convolutions.
    ~8x fewer operations than standard convs.
    Pretrained on ImageNet.

    Input:  (B, 64, H, W) from Stem
    Outputs:
        e1: (B, 16,  H/2,  W/2)
        e2: (B, 24,  H/4,  W/4)
        e3: (B, 32,  H/8,  W/8)
        e4: (B, 96,  H/16, W/16)
        e5: (B, 320, H/32, W/32)

    Speed: ~4x faster than ResNet18 on CPU
    Params: 2.2M encoder (vs 11M ResNet18 encoder)
    """
    channels = [16, 24, 32, 96, 320]

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base    = tvm.mobilenet_v2(weights=weights)

        # MobileNetV2 features are in base.features
        # Patch first conv to accept 64ch from stem
        base.features[0][0] = nn.Conv2d(
            64, 32, kernel_size=3,
            stride=2, padding=1, bias=False
        )

        # Split into encoder stages by output stride
        self.enc1 = nn.Sequential(*base.features[0:2])    # stride 2
        self.enc2 = nn.Sequential(*base.features[2:4])    # stride 4
        self.enc3 = nn.Sequential(*base.features[4:7])    # stride 8
        self.enc4 = nn.Sequential(*base.features[7:14])   # stride 16
        self.enc5 = nn.Sequential(*base.features[14:])    # stride 32

        self.proj = nn.Conv2d(1280, 320, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e5 = self.proj(e5)  # project to 320 channels
        return {
            "e1":    e1,
            "e2":    e2,
            "e3":    e3,
            "e4":    e4,
            "e5":    e5,
            "skip1": e1,
        }

# Add to models/encoder.py

class EfficientNetB0Encoder(nn.Module):
    """
    EfficientNet-B0 encoder using timm directly.
    Avoids SMP and HuggingFace cache issues entirely.
    
    Input:  (B, 64, H, W) from Stem
    Outputs 5 feature maps at different scales.
    Params: ~4M encoder
    Speed:  ~2-3x faster than ResNet18 on CPU
    """
    channels = [16, 24, 40, 112, 320]

    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        base = timm.create_model(
            "efficientnet_b0",
            pretrained   = pretrained,
            features_only= True,          # returns feature maps directly
            out_indices  = (0,1,2,3,4),   # all 5 stages
        )

        # Patch first conv to accept 64ch from Stem
        # Original: Conv2d(3, 32, ...)
        # Patched:  Conv2d(64, 32, ...)
        base.conv_stem = nn.Conv2d(
            64, 32,
            kernel_size=3, stride=2,
            padding=1, bias=False
        )

        self.base = base

    def forward(self, x):
        features = self.base(x)   # list of 5 feature maps
        e1, e2, e3, e4, e5 = features
        return {
            "e1":    e1,
            "e2":    e2,
            "e3":    e3,
            "e4":    e4,
            "e5":    e5,
            "skip1": e1,
        }

# ------------------------------------------------------------------
# Registry — add new backbones here
# ------------------------------------------------------------------

ENCODER_REGISTRY = {
    "resnet18":      ResNet18Encoder,
    "convnext_tiny": ConvNeXtTinyEncoder,
    "mobilenet_v2":  MobileNetV2Encoder,
    "efficientnet_b0": EfficientNetB0Encoder,
    # Future:
    # "swin_tiny":   SwinTinyEncoder,     → use EigenCAM
    # "segformer_b2": SegFormerB2Encoder  → use EigenCAM
}


def build_encoder(backbone, pretrained=True):
    if backbone not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown backbone: '{backbone}'. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    encoder = ENCODER_REGISTRY[backbone](pretrained=pretrained)
    print(f"Encoder: {backbone} | "
          f"pretrained={pretrained} | "
          f"channels={encoder.channels}")
    return encoder
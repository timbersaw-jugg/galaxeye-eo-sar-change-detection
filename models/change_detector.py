"""
models/change_detector.py — Full Model Assembly

Assembles:
    Stem      (models/decoder.py)
    Encoder   (models/encoder.py)
    Decoder   (models/decoder.py)

And registers GradCAM target layer.
"""

import torch
import torch.nn as nn
from models.encoder import build_encoder
from models.decoder import Stem, build_decoder


class ChangeDetector(nn.Module):
    """
    EO-SAR Binary Change Detector.

    Full pipeline:
        Input (B, 5, H, W)
            ↓ Stem
        (B, 64, H, W)
            ↓ Encoder [ResNet18 or ConvNeXt-Tiny]
        feature maps at 5 scales
            ↓ ChangeAttention at bottleneck
            ↓ UNet Decoder with skip connections
        (B, 1, H, W) logits
            ↓ Sigmoid at inference
        (B, 1, H, W) probability map 0-1
            ↓ Threshold (0.45)
        (B, 1, H, W) binary change mask

    Input channels (5):
        0: EO Red
        1: EO Green
        2: EO Blue
        3: SAR CLAHE enhanced
        4: SAR Log transformed

    Output:
        Raw logits during training (BCEWithLogitsLoss handles sigmoid)
        Sigmoid probabilities at inference
        Binary mask after thresholding

    GradCAM:
        Target layer: self.gradcam_layer = decoder.dec1.conv[1]
        CNN backbones: use GradCAMPlusPlus or LayerCAM
        register_full_backward_hook — NOT register_backward_hook
        Detach gradients/activations AFTER backward, NOT in hooks
        See utils/gradcam.py for implementation
    """

    def __init__(self, backbone="resnet18",
                 in_channels=5, pretrained=True):
        super().__init__()
        self.backbone_name = backbone

        # Stem: in_channels → 64
        self.stem    = Stem(in_channels, out_channels=64)

        # Encoder: 64ch → 5 feature maps
        self.encoder = build_encoder(backbone, pretrained)

        # Decoder: feature maps → (B, 1, H, W) logits
        self.decoder = build_decoder(self.encoder.channels)

        # GradCAM target — last ConvBNReLU in dec1 (before Dropout)
        # index 1 = second ConvBNReLU in dec1.conv sequential
        self.gradcam_layer = self.decoder.dec1.conv[1]

    def forward(self, x):
        input_size = x.shape[2:]            # (H, W)
        s          = self.stem(x)            # (B, 64, H, W)
        features   = self.encoder(s)         # dict: e1..e5, skip1
        logits     = self.decoder(features, input_size)
        return logits                        # (B, 1, H, W)

    @torch.no_grad()
    def predict(self, x, threshold=0.45):
        """
        Inference mode.
        Returns binary mask (B, 1, H, W) as uint8 {0, 1}.
        """
        self.eval()
        logits = self.forward(x)
        probs  = torch.sigmoid(logits)
        return (probs >= threshold).to(torch.uint8)


def build_model(cfg):
    model = ChangeDetector(
        backbone    = cfg["model"]["backbone"],
        in_channels = cfg["model"]["in_channels"],
        # pretrained  = cfg["model"]["pretrained"],
        pretrained = False,
    )
    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters()
                 if p.requires_grad)
    print(f"Model ready | "
          f"backbone={cfg['model']['backbone']} | "
          f"in_ch={cfg['model']['in_channels']} | "
          f"total={total/1e6:.2f}M | "
          f"trainable={train_/1e6:.2f}M")
    return model
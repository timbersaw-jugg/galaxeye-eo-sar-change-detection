"""
models/decoder.py — UNet-style Decoder with Change Attention

Components:
    Stem           : adapts any input channel count → 64ch
    ChangeAttention: channel SE attention at bottleneck
    DecoderBlock   : upsample + skip concat + 2x ConvBNReLU
    UNetDecoder    : assembles 4 DecoderBlocks + final head

Design rationale:
    Skip connections recover spatial detail lost during encoding.
    Without skips, decoder must hallucinate fine boundaries.
    With skips, decoder reconstructs exact pixel positions.
    Critical for small building footprints (~10-50px).

GradCAM target:
    self.dec1.conv[1]  ← last ConvBNReLU before Dropout in dec1
    Registered as model.gradcam_layer in ChangeDetector
    Use register_full_backward_hook (NOT register_backward_hook)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Standard Conv2d → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel,
                      padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    UNet decoder block.

    Operation:
        1. ConvTranspose2d upsample (2x)
        2. Bilinear interpolate if size mismatch (odd dims)
        3. Concatenate skip connection
        4. 2x ConvBNReLU
        5. Dropout2d(0.1) for regularization

    Args:
        in_ch:   input channels from previous decoder stage
        skip_ch: channels from encoder skip connection
        out_ch:  output channels
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                        kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch // 2 + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
            nn.Dropout2d(0.1),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------
# Channel attention at bottleneck
# ------------------------------------------------------------------

class ChangeAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention at bottleneck.
    Reweights feature channels by their change-relevance.

    Math:
        z   = GlobalAvgPool(x)                  shape: (B, C)
        w1  = ReLU(FC1(z))                      shape: (B, C//reduction)
        att = Sigmoid(FC2(w1))                  shape: (B, C)
        out = x * att.unsqueeze(-1).unsqueeze(-1)

    Reduction=8 gives good accuracy/param tradeoff.
    Adds ~0.1M params. Measured F1 gain ~0.02-0.05.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        att = self.gap(x).view(b, c)
        att = self.fc(att).view(b, c, 1, 1)
        return x * att


# ------------------------------------------------------------------
# Stem: channel adapter
# ------------------------------------------------------------------

class Stem(nn.Module):
    """
    Adapts any input channel count to 64ch for backbone.
    Handles 4ch, 5ch, 6ch inputs uniformly.

    in_channels=5: [EO_R, EO_G, EO_B, SAR_CLAHE, SAR_LOG]
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------------
# Full UNet Decoder
# ------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """
    4-stage UNet decoder.
    Takes encoder feature dict and reconstructs spatial resolution.

    enc_channels: list of [skip1_ch, e2_ch, e3_ch, e4_ch, e5_ch]
                  from encoder.channels attribute

    Output: (B, 1, H, W) raw logits — sigmoid at inference

    GradCAM target: self.dec1.conv[1]
        Register hook on this layer AFTER building decoder.
        See models/change_detector.py for hook registration.
    """
    def __init__(self, enc_channels):
        super().__init__()
        c = enc_channels  # [64, 64, 128, 256, 512] for ResNet18

        self.attn     = ChangeAttention(c[4], reduction=8)

        self.dec4     = DecoderBlock(c[4], c[3], 256)
        self.dec3     = DecoderBlock(256,   c[2], 128)
        self.dec2     = DecoderBlock(128,   c[1],  64)
        self.dec1     = DecoderBlock(64,    c[0],  32)

        self.up_final = nn.ConvTranspose2d(32, 32,
                                            kernel_size=2, stride=2)

        # Segmentation head — outputs logits (no sigmoid)
        self.head = nn.Sequential(
            ConvBNReLU(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, features, input_size):
        """
        features:   dict with keys e1..e5, skip1
        input_size: (H, W) of original input for final upsample
        """
        e5   = self.attn(features["e5"])

        d4   = self.dec4(e5,          features["e4"])
        d3   = self.dec3(d4,          features["e3"])
        d2   = self.dec2(d3,          features["e2"])
        d1   = self.dec1(d2,          features["skip1"])

        out  = self.up_final(d1)
        out  = F.interpolate(out, size=input_size,
                             mode="bilinear", align_corners=False)
        return self.head(out)                       # (B, 1, H, W)


def build_decoder(enc_channels):
    decoder = UNetDecoder(enc_channels)
    params  = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder: UNet | params={params/1e6:.2f}M | "
          f"enc_channels={enc_channels}")
    return decoder
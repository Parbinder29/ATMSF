"""
Segmentation Head for AFTMSF System
Architecture: DeepLabV3+ style with ASPP
Input: fused aggregated feature from ACAF module
Output: per-pixel class scores (B, num_classes, H, W)

Why DeepLabV3+?
    - ASPP captures multi-scale context efficiently
    - Standard benchmark in IEEE TITS segmentation papers
    - Works well with 256-channel FPN features
    - mIoU is the standard metric — directly comparable to literature

For paper: cite Chen et al. (2018) DeepLabV3+ as seg head baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    """Single ASPP branch — dilated conv at one rate."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,
                      padding=dilation, dilation=dilation,
                      bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
'''
class ASPPConv(nn.Module):
    """Single ASPP branch — dilated conv at one rate."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,
                      padding=dilation, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch of ASPP."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        out  = self.block(x)
        return F.interpolate(out, size=size,
                             mode='bilinear',
                             align_corners=False)
'''

class ASPPPooling(nn.Module):
    """Global average pooling branch of ASPP."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.conv    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm    = nn.GroupNorm(min(32, out_ch), out_ch)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        out  = self.pool(x)
        out  = self.conv(out)
        out  = self.norm(out)
        out  = self.relu(out)
        return F.interpolate(out, size=size,
                             mode='bilinear',
                             align_corners=False)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures context at multiple scales using dilated convolutions.

    Branches:
        1x1 conv          — fine detail
        3x3 dilation=6    — medium context
        3x3 dilation=12   — large context
        3x3 dilation=18   — very large context
        global avg pool   — image-level context
    """

    def __init__(
        self,
        in_channels : int = 256,
        out_channels: int = 256,
        dilations   : tuple = (3, 6),
    ):
        super().__init__()

        # 1x1 conv branch
        '''
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )'''
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated conv branches
        self.dilated = nn.ModuleList([
            ASPPConv(in_channels, out_channels, d)
            for d in dilations
        ])

        # Global pooling branch
        self.global_pool = ASPPPooling(in_channels, out_channels)

        # Project concatenated branches → out_channels
        n_branches = 1 + len(dilations) + 1   # 1x1 + dilated + pool
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_branches,
                      out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        '''
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_branches,
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )'''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.conv1x1(x)]
        for d_conv in self.dilated:
            branches.append(d_conv(x))
        branches.append(self.global_pool(x))

        out = torch.cat(branches, dim=1)
        return self.project(out)


class SegmentationHead(nn.Module):
    """
    DeepLabV3+-style segmentation head.

    Uses the 'aggregated' feature from ACAFModule as input
    (all FPN scales merged into one at highest resolution).

    Pipeline:
        aggregated feat (B,256,H/4,W/4)
            → ASPP
            → decoder convs
            → upsample 4x
            → per-pixel class scores (B, num_classes, H, W)

    Args:
        in_channels  : channels from ACAF aggregated (256)
        num_classes  : segmentation classes
        hidden_ch    : internal channels (128 — memory efficient)
        output_stride: factor to upsample final output (4)
    """

    def __init__(
        self,
        in_channels  : int = 256,
        num_classes  : int = 2,        # person + background
        hidden_ch    : int = 128,
        output_stride: int = 4,
    ):
        super().__init__()

        self.output_stride = output_stride
        self.num_classes   = num_classes

        # ASPP module
        self.aspp = ASPP(
            in_channels=in_channels,
            out_channels=hidden_ch,
            dilations=(3, 6, 12),   # smaller dilations = less memory
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_ch, hidden_ch, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        self.classifier = nn.Conv2d(hidden_ch, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(
        self,
        fused_features : dict,
        target_size    : tuple = None,
    ) -> torch.Tensor:
        """
        Args:
            fused_features : ACAFModule output dict
                             must contain 'aggregated' key
            target_size    : (H, W) of original input image
                             if None, output at stride-4 resolution

        Returns:
            seg_logits : (B, num_classes, H, W)
                         if target_size given: full resolution
                         else: H/4, W/4
        """
        feat = fused_features['aggregated']     # (B, 256, H/4, W/4)

        # ASPP
        feat = self.aspp(feat)                  # (B, 128, H/4, W/4)

        # Decoder
        feat = self.decoder(feat)               # (B, 128, H/4, W/4)

        # Classify
        logits = self.classifier(feat)          # (B, C, H/4, W/4)

        # Upsample to target resolution
        if target_size is not None:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing SegmentationHead on {device}")

    head = SegmentationHead(
        in_channels=256,
        num_classes=2,      # person + background
        hidden_ch=128,
    ).to(device)

    print(f"Trainable parameters: {head.count_parameters():,}")

    # Simulate ACAF aggregated output
    fused = {
        'aggregated': torch.randn(4, 256, 128, 160).to(device),
        '0': torch.randn(4, 256, 128, 160).to(device),
        '1': torch.randn(4, 256,  64,  80).to(device),
        '2': torch.randn(4, 256,  32,  40).to(device),
        '3': torch.randn(4, 256,  16,  20).to(device),
    }

    # Test at stride-4 resolution
    with torch.no_grad():
        logits_s4 = head(fused, target_size=None)
    print(f"Output at stride-4 : {tuple(logits_s4.shape)}")

    # Test upsampled to full resolution
    with torch.no_grad():
        logits_full = head(fused, target_size=(512, 640))
    print(f"Output full res    : {tuple(logits_full.shape)}")

    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated(device)/1024**2
        print(f"VRAM used: {mem:.1f} MB")

    print("SegmentationHead OK")
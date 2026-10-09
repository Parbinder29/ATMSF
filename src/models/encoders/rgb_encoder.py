"""
RGB Encoder for AFTMSF System
Backbone: ResNet50 with FPN neck
Optimised for 6GB VRAM via gradient checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork


class RGBEncoder(nn.Module):
    """
    ResNet50 backbone with FPN neck for RGB modality.

    Output: multi-scale feature maps at strides 4, 8, 16, 32
    Output channels: 256 at all scales (FPN normalises this)

    Args:
        pretrained      : use ImageNet pretrained weights
        out_channels    : FPN output channels (default 256)
        grad_checkpoint : use gradient checkpointing to save VRAM
        freeze_bn       : freeze BatchNorm (recommended when batch_size <= 4)
    """

    def __init__(
        self,
        pretrained     : bool = True,
        out_channels   : int  = 256,
        grad_checkpoint: bool = True,
        freeze_bn      : bool = True,
    ):
        super().__init__()

        self.out_channels    = out_channels
        self.grad_checkpoint = grad_checkpoint

        # ── Backbone ──────────────────────────────────────────────────
        weights  = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Extract layer groups — we tap after each residual stage
        self.stem   = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )                           # stride 4,  channels 64
        self.layer1 = backbone.layer1   # stride 4,  channels 256
        self.layer2 = backbone.layer2   # stride 8,  channels 512
        self.layer3 = backbone.layer3   # stride 16, channels 1024
        self.layer4 = backbone.layer4   # stride 32, channels 2048

        # ── FPN neck ──────────────────────────────────────────────────
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=out_channels,
            norm_layer=nn.BatchNorm2d,
        )

        # ── Optional BatchNorm freeze ─────────────────────────────────
        if freeze_bn:
            self._freeze_bn()

    # ------------------------------------------------------------------
    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    def _forward_backbone(self, x: torch.Tensor):
        """Forward through backbone, return feature maps at 4 scales."""
        x  = self.stem(x)
        c2 = self.layer1(x)    # stride 4
        c3 = self.layer2(c2)   # stride 8
        c4 = self.layer3(c3)   # stride 16
        c5 = self.layer4(c4)   # stride 32
        return c2, c3, c4, c5

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, H, W) normalised RGB tensor

        Returns:
            dict with keys '0','1','2','3' → feature maps
            shapes (B, 256, H/4, W/4) ... (B, 256, H/32, W/32)
        """
        if self.grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            c2, c3, c4, c5 = checkpoint(
                self._forward_backbone, x,
                use_reentrant=False
            )
        else:
            c2, c3, c4, c5 = self._forward_backbone(x)

        # FPN expects ordered dict
        features = {
            '0': c2,
            '1': c3,
            '2': c4,
            '3': c5,
        }

        return self.fpn(features)   # same keys, all 256ch


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing RGBEncoder on {device}")

    model = RGBEncoder(
        pretrained=True,
        out_channels=256,
        grad_checkpoint=True,
        freeze_bn=True,
    ).to(device)

    # Simulate batch of 4 images at KAIST resolution
    dummy = torch.randn(4, 3, 512, 640).to(device)

    with torch.no_grad():
        out = model(dummy)

    print("Output feature maps:")
    for k, v in out.items():
        print(f"  scale '{k}': {tuple(v.shape)}")

    # VRAM usage
    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated(device) / 1024**2
        print(f"VRAM used: {mem:.1f} MB")

    print("RGBEncoder OK")
"""
Thermal Encoder for AFTMSF System
Backbone: ResNet50 with FPN neck — adapted for single-channel input
Shares architecture with RGB encoder but handles 1-channel thermal images
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork


class ThermalEncoder(nn.Module):
    """
    ResNet50 backbone adapted for 1-channel thermal infrared input.

    Key difference from RGBEncoder:
        - First conv layer modified: 3-channel → 1-channel input
        - Pretrained weights averaged across RGB channels for conv1
          (standard practice — He et al., channel averaging)
        - All other layers identical to RGB encoder

    Args:
        pretrained      : use ImageNet pretrained weights (channel-averaged)
        out_channels    : FPN output channels (default 256)
        grad_checkpoint : gradient checkpointing to save VRAM
        freeze_bn       : freeze BatchNorm layers
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

        # ── Load backbone ─────────────────────────────────────────────
        weights  = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # ── Adapt conv1 for 1-channel input ───────────────────────────
        old_conv = backbone.conv1          # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = 64,
            kernel_size  = 7,
            stride       = 2,
            padding      = 3,
            bias         = False,
        )

        if pretrained:
            # Average pretrained RGB weights across channel dimension
            # Shape: (64, 3, 7, 7) → mean over dim=1 → (64, 1, 7, 7)
            new_conv.weight.data = old_conv.weight.data.mean(
                dim=1, keepdim=True
            )

        backbone.conv1 = new_conv

        # ── Extract layer groups ───────────────────────────────────────
        self.stem = nn.Sequential(
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
        x  = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 1, H, W) normalised thermal tensor

        Returns:
            dict with keys '0','1','2','3' → feature maps
            all (B, 256, H/s, W/s) where s = 4,8,16,32
        """
        if self.grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            c2, c3, c4, c5 = checkpoint(
                self._forward_backbone, x,
                use_reentrant=False
            )
        else:
            c2, c3, c4, c5 = self._forward_backbone(x)

        features = {
            '0': c2,
            '1': c3,
            '2': c4,
            '3': c5,
        }

        return self.fpn(features)


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ThermalEncoder on {device}")

    model = ThermalEncoder(
        pretrained=True,
        out_channels=256,
        grad_checkpoint=True,
        freeze_bn=True,
    ).to(device)

    # 1-channel thermal input at KAIST resolution
    dummy = torch.randn(4, 1, 512, 640).to(device)

    with torch.no_grad():
        out = model(dummy)

    print("Output feature maps:")
    for k, v in out.items():
        print(f"  scale '{k}': {tuple(v.shape)}")

    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated(device) / 1024**2
        print(f"VRAM used: {mem:.1f} MB")

    print("ThermalEncoder OK")
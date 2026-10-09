"""
Detection Head for AFTMSF System
Anchor-free detection head (FCOS-style)
Input: fused multi-scale features from ACAF module
Output: bounding boxes + class scores per FPN scale

Why FCOS-style (anchor-free)?
    - No anchor hyperparameter tuning needed
    - Works well with multi-scale FPN features
    - Standard in recent IEEE TITS papers
    - Compatible with KAIST + KITTI annotations

For paper: cite FCOS (Tian et al., 2019) as detection head baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    FCOS-style anchor-free detection head.

    For each FPN scale produces:
        - Classification scores : (B, num_classes, H, W)
        - Box regression        : (B, 4, H, W) — (l,t,r,b) offsets
        - Centerness            : (B, 1, H, W) — quality score

    Args:
        in_channels  : input channels from ACAF (256)
        num_classes  : number of foreground classes
        num_convs    : number of shared conv layers (4)
        prior_prob   : prior probability for classification bias init
    """

    def __init__(
        self,
        in_channels : int   = 256,
        num_classes : int   = 1,
        num_convs   : int   = 2,
        prior_prob  : float = 0.01,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ── Shared conv tower (cls branch) ────────────────────────────
        cls_convs = []
        for _ in range(num_convs):
            cls_convs += [
                nn.Conv2d(in_channels, in_channels, 3,
                          padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
        self.cls_tower = nn.Sequential(*cls_convs)

        # ── Shared conv tower (reg branch) ────────────────────────────
        reg_convs = []
        for _ in range(num_convs):
            reg_convs += [
                nn.Conv2d(in_channels, in_channels, 3,
                          padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
        self.reg_tower = nn.Sequential(*reg_convs)

        # ── Output predictors ─────────────────────────────────────────
        self.cls_pred    = nn.Conv2d(in_channels, num_classes, 3,
                                     padding=1)
        self.reg_pred    = nn.Conv2d(in_channels, 4, 3,
                                     padding=1)
        self.center_pred = nn.Conv2d(in_channels, 1, 3,
                                     padding=1)

    # Learnable scale factors per FPN level
        self.scales = nn.ParameterDict({
            s: nn.Parameter(torch.ones(1))
            for s in ['0', '1', '2', '3']
        })

        self._init_weights(prior_prob)
        
        # Wrap as module so they're registered
        self.scales = nn.ParameterDict({
            s: nn.Parameter(torch.ones(1))
            for s in ['0', '1', '2', '3']
        })

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        # Bias init for classification — focal loss style
        bias_val = -torch.log(
            torch.tensor((1 - prior_prob) / prior_prob)
        ).item()
        nn.init.constant_(self.cls_pred.bias, bias_val)
        nn.init.constant_(self.reg_pred.bias, 0.0)
        nn.init.constant_(self.center_pred.bias, 0.0)

        for m in [self.cls_tower, self.reg_tower]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)

    def forward_single(
        self,
        feat : torch.Tensor,
        scale: str,
    ):
        """
        Forward for a single FPN scale.

        Returns:
            cls_out    : (B, num_classes, H, W) — raw logits
            reg_out    : (B, 4, H, W)           — positive offsets
            center_out : (B, 1, H, W)           — centerness logits
        """
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)

        cls_out    = self.cls_pred(cls_feat)
        center_out = self.center_pred(cls_feat)

        # Scale regression output — each FPN level has its own scale
        reg_out = F.relu(
            self.reg_pred(reg_feat) * self.scales[scale].exp()
        )

        return cls_out, reg_out, center_out

    def forward(self, fused_features: dict):
        """
        Args:
            fused_features : output of ACAFModule
                             keys: '0','1','2','3','aggregated'

        Returns:
            predictions : list of dicts, one per FPN scale
                Each dict: {
                    'cls'    : (B, num_classes, H, W),
                    'reg'    : (B, 4, H, W),
                    'center' : (B, 1, H, W),
                    'scale'  : scale key string
                }
        """
        predictions = []
        for scale in ['0', '1', '2', '3']:
            if scale not in fused_features:
                continue
            feat = fused_features[scale]
            cls_out, reg_out, center_out = self.forward_single(
                feat, scale
            )
            predictions.append({
                'cls'   : cls_out,
                'reg'   : reg_out,
                'center': center_out,
                'scale' : scale,
            })

        return predictions

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing DetectionHead on {device}")

    head = DetectionHead(
        in_channels=256,
        num_classes=1,      # KAIST: person only
        num_convs=4,
    ).to(device)

    print(f"Trainable parameters: {head.count_parameters():,}")

    # Simulate ACAF output
    fused = {
        '0': torch.randn(4, 256, 128, 160).to(device),
        '1': torch.randn(4, 256,  64,  80).to(device),
        '2': torch.randn(4, 256,  32,  40).to(device),
        '3': torch.randn(4, 256,  16,  20).to(device),
    }

    with torch.no_grad():
        preds = head(fused)

    print(f"Predictions at {len(preds)} FPN scales:")
    for p in preds:
        print(f"  scale '{p['scale']}': "
              f"cls={tuple(p['cls'].shape)}  "
              f"reg={tuple(p['reg'].shape)}  "
              f"center={tuple(p['center'].shape)}")

    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated(device)/1024**2
        print(f"VRAM used: {mem:.1f} MB")

    print("DetectionHead OK")
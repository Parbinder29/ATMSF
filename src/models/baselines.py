"""
Baseline Models for Comparison
IEEE TITS paper — Section 4 (Experiments)

Baselines implemented:
    1. EarlyFusion  — concatenate RGB+Thermal before encoding
    2. LateFusion   — average predictions from separate models
    3. RGBOnly      — single modality RGB baseline
    4. ThermalOnly  — single modality Thermal baseline

These are the comparison models for paper Table 2.
AFTMSF must outperform all baselines to justify the contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork


class SimpleEncoder(nn.Module):
    """Shared lightweight encoder for baselines."""

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        weights   = ResNet50_Weights.IMAGENET1K_V2
        backbone  = resnet50(weights=weights)

        if in_channels != 3:
            old_conv       = backbone.conv1
            new_conv       = nn.Conv2d(in_channels, 64, 7,
                                       stride=2, padding=3, bias=False)
            new_conv.weight.data = old_conv.weight.data.mean(
                dim=1, keepdim=True
            ).repeat(1, in_channels, 1, 1) / in_channels
            backbone.conv1 = new_conv

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=out_channels,
            norm_layer=nn.BatchNorm2d,
        )

    def forward(self, x):
        x  = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn({'0':c2,'1':c3,'2':c4,'3':c5})


class SimpleDetHead(nn.Module):
    """Simple detection head for baselines."""

    def __init__(self, in_ch=256, num_classes=1):
        super().__init__()
        self.cls  = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.reg  = nn.Conv2d(in_ch, 4, 3, padding=1)
        self.ctr  = nn.Conv2d(in_ch, 1, 3, padding=1)

    def forward(self, feats):
        preds = []
        for s in ['0','1','2','3']:
            if s not in feats:
                continue
            f = feats[s]
            preds.append({
                'cls'   : self.cls(f),
                'reg'   : F.relu(self.reg(f)),
                'center': self.ctr(f),
                'scale' : s,
            })
        return preds


class SimpleSegHead(nn.Module):
    """Simple segmentation head for baselines."""

    def __init__(self, in_ch=256, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.GroupNorm(min(32,64), 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, feats, target_size=None):
        x = feats['aggregated'] if 'aggregated' in feats \
            else feats['0']
        out = self.conv(x)
        if target_size:
            out = F.interpolate(out, size=target_size,
                               mode='bilinear', align_corners=False)
        return out


class EarlyFusionModel(nn.Module):
    """
    Early Fusion Baseline.
    Concatenates RGB (3ch) + Thermal (1ch) = 4ch input.
    Single encoder processes fused input.
    No fault detection, no adaptive weighting.
    """

    def __init__(self, num_det_classes=1, num_seg_classes=2):
        super().__init__()
        self.encoder  = SimpleEncoder(in_channels=4, out_channels=256)

        # Scale aggregation
        self.agg = nn.Sequential(
            nn.Conv2d(256*4, 256, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.det_head = SimpleDetHead(256, num_det_classes)
        self.seg_head = SimpleSegHead(256, num_seg_classes)

    def forward(self, batch, target_size=None):
        rgb     = batch['rgb']
        thermal = batch['thermal']

        # Concatenate along channel dim
        x     = torch.cat([rgb, thermal], dim=1)   # (B,4,H,W)
        feats = self.encoder(x)

        # Aggregate FPN scales
        H0, W0 = feats['0'].shape[-2:]
        ups = []
        for s in ['0','1','2','3']:
            f = feats[s]
            if f.shape[-2:] != (H0, W0):
                f = F.interpolate(f, (H0,W0),
                                  mode='bilinear',
                                  align_corners=False)
            ups.append(f)
        feats['aggregated'] = self.agg(torch.cat(ups, dim=1))

        return {
            'det_preds' : self.det_head(feats),
            'seg_logits': self.seg_head(feats, target_size),
            'confidence': {'rgb': torch.ones(rgb.shape[0]),
                          'thermal': torch.ones(rgb.shape[0])},
            'fault_flags': {},
            'acaf_weights': {},
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class LateFusionModel(nn.Module):
    """
    Late Fusion Baseline.
    Separate encoders for RGB and Thermal.
    Averages final predictions (equal weights, no adaptation).
    No fault detection mechanism.
    """

    def __init__(self, num_det_classes=1, num_seg_classes=2):
        super().__init__()
        self.rgb_enc = SimpleEncoder(in_channels=3, out_channels=256)
        self.thr_enc = SimpleEncoder(in_channels=1, out_channels=256)

        self.agg = nn.Sequential(
            nn.Conv2d(256*4, 256, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.det_head = SimpleDetHead(256, num_det_classes)
        self.seg_head = SimpleSegHead(256, num_seg_classes)

    def _aggregate(self, feats):
        H0, W0 = feats['0'].shape[-2:]
        ups = []
        for s in ['0','1','2','3']:
            f = feats[s]
            if f.shape[-2:] != (H0,W0):
                f = F.interpolate(f,(H0,W0),
                                  mode='bilinear',
                                  align_corners=False)
            ups.append(f)
        return self.agg(torch.cat(ups, dim=1))

    def forward(self, batch, target_size=None):
        rgb     = batch['rgb']
        thermal = batch['thermal']

        rgb_feats = self.rgb_enc(rgb)
        thr_feats = self.thr_enc(thermal)

        # Average features (equal weight late fusion)
        fused = {}
        for s in ['0','1','2','3']:
            fused[s] = (rgb_feats[s] + thr_feats[s]) / 2.0
        fused['aggregated'] = self._aggregate(fused)

        return {
            'det_preds' : self.det_head(fused),
            'seg_logits': self.seg_head(fused, target_size),
            'confidence': {'rgb': torch.ones(rgb.shape[0]),
                          'thermal': torch.ones(rgb.shape[0])},
            'fault_flags': {},
            'acaf_weights': {},
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class RGBOnlyModel(nn.Module):
    """RGB-only single modality baseline."""

    def __init__(self, num_det_classes=1, num_seg_classes=2):
        super().__init__()
        self.encoder  = SimpleEncoder(in_channels=3, out_channels=256)
        self.agg = nn.Sequential(
            nn.Conv2d(256*4, 256, 1, bias=False),
            nn.GroupNorm(32, 256), nn.GELU(),
        )
        self.det_head = SimpleDetHead(256, num_det_classes)
        self.seg_head = SimpleSegHead(256, num_seg_classes)

    def forward(self, batch, target_size=None):
        rgb   = batch['rgb']
        feats = self.encoder(rgb)
        H0, W0 = feats['0'].shape[-2:]
        ups = [F.interpolate(feats[s],(H0,W0),
               mode='bilinear',align_corners=False)
               if feats[s].shape[-2:]!=(H0,W0)
               else feats[s] for s in ['0','1','2','3']]
        feats['aggregated'] = self.agg(torch.cat(ups,dim=1))
        return {
            'det_preds' : self.det_head(feats),
            'seg_logits': self.seg_head(feats, target_size),
            'confidence': {}, 'fault_flags': {},
            'acaf_weights': {},
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
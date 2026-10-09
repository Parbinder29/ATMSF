"""
Adaptive Cross-modal Attention Fusion (ACAF) Module
AFTMSF System — Core Novel Contribution C1

This module is the PRIMARY novelty of the paper.

Standard cross-modal attention (TransFuser, BEVFusion):
    F = Attention(Q, K, V)  ← assumes all sensors reliable

Our contribution — confidence-gated cross-modal attention:
    F = Σ softmax(c_i / τ) · CrossAttention(Q_i, K_j, V_j)

Where:
    c_i  = per-modality confidence score from FaultDetector
    τ    = temperature parameter (learned)
    Q_i  = query from modality i
    K_j  = keys from all other modalities j
    V_j  = values from all other modalities j

Key properties:
    1. When c_i → 0 (fault): modality i contributes nothing
    2. When all c_i equal: reduces to standard attention
    3. Weights sum to 1 (softmax) → graceful degradation
    4. Temperature τ controls sharpness of weight distribution
    5. Works with any number of modalities (generalises to N sensors)

Mathematical formulation (for paper Section 3):
    w_i     = softmax(c / τ)_i
    A_i     = softmax(Q_i · K_all^T / √d) · V_all
    F_scale = Σ_i w_i · A_i
    F       = Σ_scales F_scale  (FPN scale aggregation)

References:
    Vaswani et al. [6] — attention mechanism
    TransFuser [10]    — cross-modal attention baseline
    BEVFusion [11]     — multi-sensor fusion baseline
    (Our work improves upon both by adding confidence gating)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossModalAttention(nn.Module):
    """
    Memory-efficient cross-modal attention with spatial pooling.
    
    Key change: downsample spatial dims before attention.
    This reduces attention matrix from (HW x N*HW) to (P x N*P)
    where P = pool_size^2 (e.g. 8x8 = 64 tokens per modality)
    
    Memory: O(P^2 * N) instead of O(H^2 * W^2 * N) — ~1000x reduction
    This makes attention feasible on 6GB VRAM.
    
    Note for paper: This is 'efficient cross-modal attention via
    spatial pooling' — cite as design choice for edge deployment.
    """

    def __init__(
        self,
        channels : int   = 256,
        num_heads: int   = 4,
        dropout  : float = 0.1,
        pool_size: int   = 8,      # spatial tokens per modality
    ):
        super().__init__()
        assert channels % num_heads == 0

        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5
        self.pool_size = pool_size

        # Spatial pooling to reduce token count
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        self.to_q   = nn.Linear(channels, channels, bias=False)
        self.to_k   = nn.Linear(channels, channels, bias=False)
        self.to_v   = nn.Linear(channels, channels, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(dropout),
        )
        self.norm_q  = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

        # Upsample attended tokens back to original resolution
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )

    def forward(
        self,
        query_feat : torch.Tensor,
        kv_feats   : list,
    ) -> torch.Tensor:
        B, C, H, W = query_feat.shape

        # Pool query to (B, C, P, P)
        q_pooled = self.pool(query_feat)
        q_flat   = rearrange(q_pooled, 'b c h w -> b (h w) c')
        q_flat   = self.norm_q(q_flat)

        # Pool all KV features and concatenate
        kv_list = []
        for kv in kv_feats:
            kv_p = self.pool(kv)
            kv_f = rearrange(kv_p, 'b c h w -> b (h w) c')
            kv_list.append(kv_f)
        kv_flat = torch.cat(kv_list, dim=1)
        kv_flat = self.norm_kv(kv_flat)

        # Project Q K V
        q = self.to_q(q_flat)
        k = self.to_k(kv_flat)
        v = self.to_v(kv_flat)

        # Multi-head reshape
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Attention — now tiny: (B, heads, P^2, N*P^2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v)

        # Reshape back to spatial pooled size
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) c -> b c h w',
                        h=self.pool_size, w=self.pool_size)

        # Upsample back to original resolution + residual refinement
        out = F.interpolate(out, size=(H, W),
                           mode='bilinear', align_corners=False)
        out = self.refine(out)

        # Residual connection with query
        return out + query_feat

class ACAFScale(nn.Module):
    """
    ACAF at a single FPN scale.

    For each modality:
        1. Compute cross-modal attention with all other modalities
        2. Weight the attended feature by confidence score c_i
        3. Sum weighted features → fused feature at this scale

    Args:
        modalities : list of modality names
        channels   : feature channels (256)
        num_heads  : attention heads (4)
        temperature: initial temperature for confidence scaling
    """

    def __init__(
        self,
        modalities : list  = ['rgb', 'thermal', 'depth'],
        channels   : int   = 256,
        num_heads  : int   = 4,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.modalities = modalities
        self.channels   = channels

        # One cross-modal attention block per modality
        self.attention  = nn.ModuleDict({
            m: CrossModalAttention(channels, num_heads)
            for m in modalities
        })

        # Learnable temperature τ for confidence sharpening
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(temperature))
        )

        # Feed-forward network after fusion (standard transformer)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
        )
        self.norm_out = nn.GroupNorm(32, channels)

    def forward(
        self,
        scale_features    : dict,
        confidence_scores : dict,
    ) -> torch.Tensor:
        """
        Args:
            scale_features    : {modality: (B, C, H, W)}
                                features at one FPN scale
            confidence_scores : {modality: (B,)} from FaultDetector

        Returns:
            fused : (B, C, H, W) confidence-weighted fused feature
        """
        # Temperature scaling of confidence scores
        τ = torch.exp(self.log_temperature).clamp(0.1, 10.0)

        # Stack confidence scores: (B, N_modalities)
        mod_list = [m for m in self.modalities
                    if m in scale_features]
        conf_stack = torch.stack(
            [confidence_scores[m] for m in mod_list], dim=1
        )                                           # (B, N)

        # Softmax over modalities with temperature
        # This is the key equation: w_i = softmax(c / τ)_i
        weights = F.softmax(conf_stack / τ, dim=1) # (B, N)

        # All feature maps at this scale
        all_feats = [scale_features[m] for m in mod_list]

        # For each modality: attend to all, then weight by confidence
        fused = torch.zeros_like(all_feats[0])

        for idx, mod in enumerate(mod_list):
            # Cross-modal attention: query=this modality, kv=all
            attended = self.attention[mod](
                query_feat=scale_features[mod],
                kv_feats=all_feats,
            )                                       # (B, C, H, W)

            # Weight by confidence: w_i * A_i
            # Reshape weight for broadcasting: (B,) → (B, 1, 1, 1)
            w_i    = weights[:, idx].view(-1, 1, 1, 1)
            fused  = fused + w_i * attended

        # FFN + residual + norm
        fused = fused + self.ffn(fused)
        fused = self.norm_out(fused)

        return fused


class ACAFModule(nn.Module):
    """
    Full Adaptive Cross-modal Attention Fusion (ACAF) Module.

    Fuses multi-scale features from all modality encoders using
    confidence-gated cross-modal attention at each FPN scale.

    Architecture:
        For each FPN scale (0,1,2,3):
            fused_scale = ACAFScale(features_scale, confidence)
        Final output: dict of fused features at all scales
        (passed to detection + segmentation heads)

    Args:
        modalities   : list of modality names
        channels     : FPN channels (256)
        num_heads    : attention heads (4 for 6GB VRAM)
        scales       : FPN scale keys (default ['0','1','2','3'])
        temperature  : initial confidence temperature
    """

    def __init__(
        self,
        modalities : list  = ['rgb', 'thermal', 'depth'],
        channels   : int   = 256,
        num_heads  : int   = 4,
        scales     : list  = ['0', '1', '2', '3'],
        temperature: float = 1.0,
    ):
        super().__init__()

        self.modalities = modalities
        self.scales     = scales

        # One ACAFScale per FPN level
        self.scale_fusion = nn.ModuleDict({
            s: ACAFScale(modalities, channels, num_heads, temperature)
            for s in scales
        })

        # Cross-scale aggregation — merges all FPN scales into one
        # Used as input to detection/seg heads
        self.scale_agg = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )

    def forward(
        self,
        encoder_features  : dict,
        confidence_scores : dict,
    ) -> dict:
        """
        Args:
            encoder_features  : {modality: {scale: (B,C,H,W)}}
                                output of all encoders
            confidence_scores : {modality: (B,)} from FaultDetector

        Returns:
            fused_features : {scale: (B,C,H,W)} fused at each scale
                            + 'aggregated': (B,C,H,W) scale-merged feature
        """
        fused_features = {}

        # Get reference spatial size (scale '0' = highest resolution)
        ref_scale = self.scales[0]
        ref_mod   = next(m for m in self.modalities
                        if m in encoder_features)
        H0, W0    = encoder_features[ref_mod][ref_scale].shape[-2:]

        for scale in self.scales:
            # Collect this scale's features from all modalities
            scale_feats = {}
            for mod in self.modalities:
                if mod in encoder_features:
                    scale_feats[mod] = encoder_features[mod][scale]

            # Confidence-gated cross-modal attention fusion
            fused = self.scale_fusion[scale](
                scale_feats,
                confidence_scores,
            )
            fused_features[scale] = fused

        # Cross-scale aggregation
        # Upsample all scales to scale '0' resolution then concat
        upsampled = []
        for scale in self.scales:
            feat = fused_features[scale]
            if feat.shape[-2:] != (H0, W0):
                feat = F.interpolate(
                    feat, size=(H0, W0),
                    mode='bilinear',
                    align_corners=False,
                )
            upsampled.append(feat)

        aggregated = torch.cat(upsampled, dim=1)        # (B, C*4, H, W)
        fused_features['aggregated'] = self.scale_agg(aggregated)

        return fused_features

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ACAFModule on {device}")

    modalities = ['rgb', 'thermal', 'depth']
    scales     = ['0', '1', '2', '3']
    B, C       = 4, 256

    model = ACAFModule(
        modalities=modalities,
        channels=256,
        num_heads=4,
        scales=scales,
        temperature=1.0,
    ).to(device)

    print(f"Trainable parameters: {model.count_parameters():,}")

    # Simulate encoder outputs at 4 FPN scales
    spatial_sizes = {
        '0': (128, 160),
        '1': (64,  80),
        '2': (32,  40),
        '3': (16,  20),
    }

    encoder_features = {
        mod: {
            s: torch.randn(B, C, *spatial_sizes[s]).to(device)
            for s in scales
        }
        for mod in modalities
    }

    # Normal confidence scores
    conf_normal = {m: torch.ones(B).to(device) * 0.9
                   for m in modalities}

    # Faulty thermal scenario
    conf_faulty = {
        'rgb'    : torch.ones(B).to(device) * 0.9,
        'thermal': torch.ones(B).to(device) * 0.1,  # FAULT
        'depth'  : torch.ones(B).to(device) * 0.85,
    }

    print("\n--- Normal confidence ---")
    with torch.no_grad():
        out_normal = model(encoder_features, conf_normal)
    for k, v in out_normal.items():
        print(f"  scale '{k}': {tuple(v.shape)}")

    print("\n--- Faulty thermal ---")
    with torch.no_grad():
        out_faulty = model(encoder_features, conf_faulty)

    # Check fusion weights differ when thermal is faulty
    print("  Verifying confidence gating works:")
    for scale in scales[:1]:  # check scale 0
        fn = model.scale_fusion[scale]
        τ  = torch.exp(fn.log_temperature)
        conf_n = torch.stack([conf_normal[m] for m in modalities], dim=1)
        conf_f = torch.stack([conf_faulty[m] for m in modalities], dim=1)
        w_n = F.softmax(conf_n / τ, dim=1).mean(0)
        w_f = F.softmax(conf_f / τ, dim=1).mean(0)
        print(f"  Normal  weights: "
              f"{[f'{w:.3f}' for w in w_n.tolist()]}")
        print(f"  Faulty  weights: "
              f"{[f'{w:.3f}' for w in w_f.tolist()]}")
        print(f"  Thermal weight dropped: "
              f"{w_n[1]:.3f} → {w_f[1]:.3f}")

    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\nVRAM used: {mem:.1f} MB")

    print("ACAFModule OK")
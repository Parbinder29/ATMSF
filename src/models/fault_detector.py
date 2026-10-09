"""
Fault Detection and Reliability Estimation Module
AFTMSF System — Core Novel Contribution C2

For each sensor modality, computes a scalar confidence score
c_i ∈ [0, 1] using THREE signals jointly:

    1. Reconstruction error  — autoencoder-based
       High recon error → sensor input is anomalous
    2. Feature variance      — statistical
       Very low variance → sensor dropout (all zeros)
       Very high variance → sensor noise
    3. Prediction entropy    — information-theoretic
       High entropy → model is uncertain about this modality

Final confidence score:
    c_i = σ( w1·(1 - recon_err) + w2·var_score + w3·(1 - entropy) )

This unified 3-signal approach is the novelty.
No prior fusion paper combines all three signals for per-modality
reliability estimation at runtime.

Reference for paper:
    - Autoencoder anomaly detection: Chalapathy & Chawla (2019)
    - Uncertainty estimation: Kendall & Gal (2017)
    - Feature variance for fault detection: Wang et al. [12]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAutoencoder(nn.Module):
    """
    Lightweight autoencoder for one modality's feature map.
    Reconstruction error signals anomalous sensor input.

    Input/Output: (B, in_channels, H, W) feature map
    Bottleneck forces compression — faults cause high recon error.
    """

    def __init__(self, in_channels: int = 256, bottleneck: int = 64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, bottleneck, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class ReliabilityEstimator(nn.Module):
    """
    Computes scalar confidence score c_i ∈ [0,1] for one modality.

    Takes the FPN feature map at scale '0' (highest resolution)
    as the reliability signal — most sensitive to input quality.

    Args:
        in_channels  : FPN output channels (default 256)
        bottleneck   : autoencoder bottleneck dim (default 64)
        num_classes  : number of task classes for entropy estimation
    """

    def __init__(
        self,
        in_channels : int = 256,
        bottleneck  : int = 64,
        num_classes : int = 2,      # person + background
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Signal 1 — Autoencoder for reconstruction error
        self.autoencoder = ModalityAutoencoder(in_channels, bottleneck)

        # Signal 3 — Small classifier head for entropy estimation
        # Predicts class distribution from features → entropy
        self.entropy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B, C, 1, 1)
            nn.Flatten(),                   # (B, C)
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # Learnable signal combination weights
        # Initialised equally, learned during training
        self.w = nn.Parameter(torch.ones(3) / 3.0)

    # ------------------------------------------------------------------
    def _reconstruction_error(
        self,
        feat   : torch.Tensor,
        recon  : torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalised MSE reconstruction error per sample.
        Returns (B,) tensor in [0, 1].
        High value = anomalous input.
        """
        err = F.mse_loss(recon, feat, reduction='none')
        # Mean over C, H, W — keep batch dim
        err = err.mean(dim=[1, 2, 3])
        # Normalise to [0,1] using sigmoid scaling
        return torch.sigmoid(err * 10.0)

    # ------------------------------------------------------------------
    def _variance_score(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Feature variance score per sample.
        Returns (B,) tensor in [0, 1].

        Normal sensor  → moderate variance (score near 1)
        Dropout fault  → near-zero variance (score near 0)
        Noise fault    → very high variance (score near 0)
        """
        # Variance over C, H, W
        var = feat.var(dim=[1, 2, 3])

        # Expected variance range for normalised features: [0.1, 2.0]
        # Score peaks at moderate variance, drops at extremes
        var_norm = var / (var + 1.0)    # maps (0,∞) → (0,1)

        # Penalise very low variance (dropout) and very high (noise)
        score = 1.0 - torch.abs(var_norm - 0.5) * 2.0
        return score.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def _entropy_score(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Prediction entropy per sample.
        Returns (B,) tensor in [0, 1].
        High entropy = high uncertainty = low reliability.
        Score = 1 - normalised_entropy (so high score = reliable)
        """
        logits = self.entropy_head(feat)        # (B, num_classes)
        probs  = F.softmax(logits, dim=-1)      # (B, num_classes)

        # Shannon entropy: H = -Σ p log p
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # (B,)

        # Normalise by max entropy (log of num_classes)
        max_entropy = torch.log(
            torch.tensor(self.num_classes, dtype=torch.float32)
        )
        norm_entropy = entropy / (max_entropy + 1e-8)

        # High entropy → low reliability score
        return (1.0 - norm_entropy).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat : (B, C, H, W) feature map from encoder FPN scale '0'

        Returns:
            confidence : (B,) scalar confidence score per sample
            aux_dict   : dict with individual signal values for logging
                         and ablation study (paper Table 3)
        """
        # Signal 1: reconstruction error
        recon, _   = self.autoencoder(feat)
        recon_err  = self._reconstruction_error(feat, recon)
        sig1       = 1.0 - recon_err           # high = reliable

        # Signal 2: feature variance
        sig2 = self._variance_score(feat)      # high = reliable

        # Signal 3: prediction entropy
        sig3 = self._entropy_score(feat)       # high = reliable

        # Weighted combination with learned weights
        # w is softmaxed so weights sum to 1
        w = F.softmax(self.w, dim=0)
        confidence = w[0]*sig1 + w[1]*sig2 + w[2]*sig3   # (B,)
        confidence = confidence.clamp(0.0, 1.0)

        aux_dict = {
            'recon_error'   : recon_err.detach(),   # (B,)
            'variance_score': sig2.detach(),         # (B,)
            'entropy_score' : sig3.detach(),         # (B,)
            'weights'       : w.detach(),            # (3,)
            'confidence'    : confidence.detach(),   # (B,)
        }

        return confidence, aux_dict


class FaultDetector(nn.Module):
    """
    Full fault detection module for all modalities.

    Wraps one ReliabilityEstimator per modality.
    Outputs per-modality confidence scores used by ACAF module
    to gate the attention fusion weights.

    This is the module described in Section 3.0 of your synopsis:
    'fault detection mechanism that continuously monitors
    the quality of sensor inputs'

    Args:
        modalities   : list of modality names
        in_channels  : FPN channels from encoders (default 256)
        num_classes  : task classes for entropy head
        fault_thresh : confidence below this = fault declared (0.3)
    """

    def __init__(
        self,
        modalities  : list = ['rgb', 'thermal', 'depth'],
        in_channels : int  = 256,
        num_classes : int  = 2,
        fault_thresh: float = 0.15,
    ):
        super().__init__()

        self.modalities   = modalities
        self.fault_thresh = fault_thresh

        # One estimator per modality
        self.estimators = nn.ModuleDict({
            m: ReliabilityEstimator(in_channels, 64, num_classes)
            for m in modalities
        })

    # ------------------------------------------------------------------
    def forward(self, features: dict):
        """
        Args:
            features : dict mapping modality name → FPN feature dict
                       e.g. {'rgb': {'0':..,'1':..},
                              'thermal': {'0':..,'1':..},
                              'depth': {'0':..,'1':..}}
                       We use scale '0' (highest resolution) for reliability

        Returns:
            confidence_scores : dict {modality: (B,) tensor}
            fault_flags       : dict {modality: (B,) bool tensor}
                                True = fault detected
            aux_outputs       : dict {modality: aux_dict} for logging
        """
        confidence_scores = {}
        fault_flags       = {}
        aux_outputs       = {}

        for mod in self.modalities:
            if mod not in features:
                # Modality missing entirely — max fault
                B = next(iter(
                    next(iter(features.values())).values()
                )).shape[0]
                confidence_scores[mod] = torch.zeros(B)
                fault_flags[mod]       = torch.ones(B, dtype=torch.bool)
                continue

            # Use highest-resolution feature map (scale '0')
            feat = features[mod]['0']

            conf, aux = self.estimators[mod](feat)

            confidence_scores[mod] = conf
            fault_flags[mod]       = conf < self.fault_thresh
            aux_outputs[mod]       = aux

        return confidence_scores, fault_flags, aux_outputs

    # ------------------------------------------------------------------
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing FaultDetector on {device}")

    detector = FaultDetector(
        modalities  =['rgb', 'thermal', 'depth'],
        in_channels =256,
        num_classes =2,
        fault_thresh=0.15,
    ).to(device)

    print(f"Trainable parameters: {detector.count_parameters():,}")

    B, C, H, W = 4, 256, 128, 160

    # Simulate normal features
    features_normal = {
        'rgb'    : {'0': torch.randn(B,C,H,W).to(device)},
        'thermal': {'0': torch.randn(B,C,H,W).to(device)},
        'depth'  : {'0': torch.randn(B,C,H,W).to(device)},
    }

    # Simulate faulty thermal (dropout — all zeros)
    features_faulty = {
        'rgb'    : {'0': torch.randn(B,C,H,W).to(device)},
        'thermal': {'0': torch.zeros(B,C,H,W).to(device)},  # FAULT
        'depth'  : {'0': torch.randn(B,C,H,W).to(device)},
    }

    print("\n--- Normal inputs ---")
    with torch.no_grad():
        conf, flags, aux = detector(features_normal)
    for mod in ['rgb','thermal','depth']:
        print(f"  {mod:8s} confidence: "
              f"{conf[mod].mean():.3f}  "
              f"fault: {flags[mod].any().item()}")

    print("\n--- Faulty thermal (dropout) ---")
    with torch.no_grad():
        conf, flags, aux = detector(features_faulty)
    for mod in ['rgb','thermal','depth']:
        print(f"  {mod:8s} confidence: "
              f"{conf[mod].mean():.3f}  "
              f"fault: {flags[mod].any().item()}")

    print("\n--- Signal weights (learned) ---")
    for mod in ['rgb','thermal','depth']:
        w = aux[mod]['weights']
        print(f"  {mod}: recon={w[0]:.3f} "
              f"variance={w[1]:.3f} "
              f"entropy={w[2]:.3f}")

    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated(device)/1024**2
        print(f"\nVRAM used: {mem:.1f} MB")

    print("FaultDetector OK")
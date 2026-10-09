"""
AFTMSF — Adaptive Fault-Tolerant Multi-Sensor Fusion Model
Full assembled model combining all modules.

Architecture flow:
    RGB   (B,3,H,W) ──► RGBEncoder     ──►┐
    Thermal(B,1,H,W) ──► ThermalEncoder ──►├──► FaultDetector
    Depth  (B,1,H,W) ──► DepthEncoder   ──►┘         │
                                                       │ confidence scores
                                                       ▼
                                               ACAFModule (fusion)
                                                       │
                               ┌───────────────────────┤
                               ▼                       ▼
                        DetectionHead         SegmentationHead
                               │                       │
                          bbox preds              seg logits

Paper section mapping:
    Section 3.0  — full system description
    Section 3.1  — problem formulation → forward() inputs/outputs
    Section 3.2  — objectives → each module's role
    Fig 1        — this exact architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.rgb_encoder     import RGBEncoder
from .encoders.thermal_encoder import ThermalEncoder
from .encoders.depth_encoder   import DepthEncoder
from .fault_detector           import FaultDetector
from .acaf_module              import ACAFModule
from .detection_head           import DetectionHead
from .seg_head                 import SegmentationHead


class AFTMSFModel(nn.Module):
    """
    Full Adaptive Fault-Tolerant Multi-Sensor Fusion Model.

    Supports flexible modality input:
        - All 3 modalities (RGB + Thermal + Depth)
        - RGB + Thermal only (KAIST)
        - RGB + Depth only   (KITTI subset)
        - Single modality    (graceful degradation test)

    Args:
        modalities    : list of active modalities
                        subset of ['rgb', 'thermal', 'depth']
        num_det_classes : detection classes (1 for KAIST person)
        num_seg_classes : segmentation classes (2 for person+bg)
        fpn_channels  : FPN output channels (256)
        num_heads     : ACAF attention heads (4)
        fault_thresh  : confidence threshold for fault declaration
        grad_checkpoint: use gradient checkpointing (recommended)
    """

    def __init__(
        self,
        modalities      : list  = ['rgb', 'thermal', 'depth'],
        num_det_classes : int   = 1,
        num_seg_classes : int   = 2,
        fpn_channels    : int   = 256,
        num_heads       : int   = 4,
        fault_thresh    : float = 0.3,
        grad_checkpoint : bool  = True,
    ):
        super().__init__()

        self.modalities      = modalities
        self.fpn_channels    = fpn_channels
        self.num_det_classes = num_det_classes
        self.num_seg_classes = num_seg_classes

        # ── Encoders (only build what we need) ────────────────────────
        self.encoders = nn.ModuleDict()

        if 'rgb' in modalities:
            self.encoders['rgb'] = RGBEncoder(
                pretrained=True,
                out_channels=fpn_channels,
                grad_checkpoint=grad_checkpoint,
                freeze_bn=True,
            )

        if 'thermal' in modalities:
            self.encoders['thermal'] = ThermalEncoder(
                pretrained=True,
                out_channels=fpn_channels,
                grad_checkpoint=grad_checkpoint,
                freeze_bn=True,
            )

        if 'depth' in modalities:
            self.encoders['depth'] = DepthEncoder(
                in_channels=1,
                out_channels=fpn_channels,
                grad_checkpoint=grad_checkpoint,
                freeze_bn=True,
            )

        # ── Fault Detector ────────────────────────────────────────────
        self.fault_detector = FaultDetector(
            modalities=modalities,
            in_channels=fpn_channels,
            num_classes=num_det_classes + 1,  # +1 for background
            fault_thresh=fault_thresh,
        )

        # ── ACAF Fusion Module ────────────────────────────────────────
        self.acaf = ACAFModule(
            modalities=modalities,
            channels=fpn_channels,
            num_heads=num_heads,
            scales=['0', '1', '2', '3'],
            temperature=1.0,
        )

        # ── Task Heads ────────────────────────────────────────────────
        self.det_head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_det_classes,
            num_convs=4,
        )

        self.seg_head = SegmentationHead(
            in_channels=fpn_channels,
            num_classes=num_seg_classes,
            hidden_ch=64,
        )

    # ------------------------------------------------------------------
    def encode(self, batch: dict) -> dict:
        """
        Run all encoders. Returns encoder_features dict.

        Args:
            batch : dict with keys matching self.modalities
                    'rgb'     : (B, 3, H, W)
                    'thermal' : (B, 1, H, W)
                    'depth'   : (B, 1, H, W)

        Returns:
            encoder_features : {modality: {scale: tensor}}
        """
        encoder_features = {}
        for mod in self.modalities:
            if mod in batch and batch[mod] is not None:
                encoder_features[mod] = self.encoders[mod](batch[mod])
        return encoder_features

    # ------------------------------------------------------------------
    def forward(
        self,
        batch       : dict,
        target_size : tuple = None,
    ) -> dict:
        """
        Full forward pass.

        Args:
            batch : dict with modality tensors
                {
                  'rgb'     : (B, 3, H, W),   optional
                  'thermal' : (B, 1, H, W),   optional
                  'depth'   : (B, 1, H, W),   optional
                }
            target_size : (H, W) for seg output upsampling
                          if None uses stride-4 resolution

        Returns:
            output dict:
            {
              'det_preds'    : list of per-scale detection dicts
              'seg_logits'   : (B, num_seg_classes, H, W)
              'confidence'   : {modality: (B,)} reliability scores
              'fault_flags'  : {modality: (B,) bool} fault indicators
              'acaf_weights' : {scale: (B, N)} fusion weights per scale
              'aux'          : auxiliary outputs for logging
            }
        """
        # Step 1 — Encode all modalities
        encoder_features = self.encode(batch)

        # Step 2 — Fault detection + reliability estimation
        confidence, fault_flags, aux_fd = self.fault_detector(
            encoder_features
        )

        # Step 3 — Confidence-gated adaptive fusion
        fused_features = self.acaf(encoder_features, confidence)

        # Step 4 — Detection head
        det_preds = self.det_head(fused_features)

        # Step 5 — Segmentation head
        seg_logits = self.seg_head(fused_features, target_size)

        # Compute fusion weights for logging/visualisation
        acaf_weights = self._get_fusion_weights(confidence)

        return {
            'det_preds'   : det_preds,
            'seg_logits'  : seg_logits,
            'confidence'  : confidence,
            'fault_flags' : fault_flags,
            'acaf_weights': acaf_weights,
            'aux'         : aux_fd,
        }

    # ------------------------------------------------------------------
    def _get_fusion_weights(self, confidence: dict) -> dict:
        """
        Compute normalised fusion weights for visualisation.
        These are logged during training and plotted in the paper.
        """
        if not confidence:
            return {}

        mod_list   = list(confidence.keys())
        conf_stack = torch.stack(
            [confidence[m] for m in mod_list], dim=1
        )   # (B, N)

        # Get temperature from first ACAF scale
        scale0 = self.acaf.scale_fusion['0']
        τ = torch.exp(scale0.log_temperature).clamp(0.1, 10.0)

        weights = torch.softmax(conf_stack / τ, dim=1)  # (B, N)

        return {
            mod: weights[:, i]
            for i, mod in enumerate(mod_list)
        }

    # ------------------------------------------------------------------
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters()
                    if p.requires_grad)
        return total, train

    # ------------------------------------------------------------------
    def set_train_stage(self, stage: int):
        """
        Training curriculum — unfreeze progressively.

        Stage 1: freeze encoders, train fusion + heads only
        Stage 2: unfreeze encoders layer4 + train all
        Stage 3: full end-to-end training

        This staged approach is standard for multi-modal papers
        and mentioned explicitly in the methodology section.
        """
        if stage == 1:
            # Freeze all encoder parameters
            for mod in self.modalities:
                for p in self.encoders[mod].parameters():
                    p.requires_grad = False
            print("[AFTMSF] Stage 1: encoders frozen, "
                  "training fusion + heads")

        elif stage == 2:
            # Unfreeze layer4 of RGB + Thermal encoders
            for mod in ['rgb', 'thermal']:
                if mod in self.encoders:
                    for p in self.encoders[mod].layer4.parameters():
                        p.requires_grad = True
            print("[AFTMSF] Stage 2: encoder layer4 unfrozen")

        elif stage == 3:
            # Full end-to-end
            for p in self.parameters():
                p.requires_grad = True
            print("[AFTMSF] Stage 3: full end-to-end training")

    # ------------------------------------------------------------------
    def get_parameter_groups(self, base_lr: float):
        """
        Parameter groups for optimizer with differential learning rates.
        Encoders use 0.1x lr, fusion/heads use 1.0x lr.
        Standard practice for fine-tuning pretrained backbones.
        """
        encoder_params = []
        other_params   = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'encoders' in name:
                encoder_params.append(p)
            else:
                other_params.append(p)

        return [
            {'params': encoder_params, 'lr': base_lr * 0.1},
            {'params': other_params,   'lr': base_lr},
        ]


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing AFTMSFModel on {device}")

    # Test with RGB + Thermal (KAIST setup)
    model = AFTMSFModel(
        modalities=['rgb', 'thermal'],
        num_det_classes=1,
        num_seg_classes=2,
        fpn_channels=256,
        num_heads=4,
        fault_thresh=0.3,
    ).to(device)

    total, trainable = model.count_parameters()
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Simulate KAIST batch
    batch = {
        'rgb'    : torch.randn(2, 3, 512, 640).to(device),
        'thermal': torch.randn(2, 1, 512, 640).to(device),
    }

    print("\n--- Forward pass (normal) ---")
    with torch.no_grad():
        out = model(batch, target_size=(512, 640))

    print(f"Detection preds : {len(out['det_preds'])} scales")
    for p in out['det_preds']:
        print(f"  scale '{p['scale']}': "
              f"cls={tuple(p['cls'].shape)}")

    print(f"Seg logits      : {tuple(out['seg_logits'].shape)}")
    print(f"Confidence      : "
          f"rgb={out['confidence']['rgb'].mean():.3f}  "
          f"thermal={out['confidence']['thermal'].mean():.3f}")

    print("\n--- Fault injection test ---")
    batch_faulty = {
        'rgb'    : torch.randn(2, 3, 512, 640).to(device),
        'thermal': torch.zeros(2, 1, 512, 640).to(device),  # FAULT
    }
    with torch.no_grad():
        out_f = model(batch_faulty, target_size=(512, 640))

    print(f"Confidence (faulty thermal):")
    print(f"  rgb     : {out_f['confidence']['rgb'].mean():.3f}")
    print(f"  thermal : {out_f['confidence']['thermal'].mean():.3f}")
    print(f"Fault flags:")
    print(f"  rgb     : {out_f['fault_flags']['rgb'].any().item()}")
    print(f"  thermal : {out_f['fault_flags']['thermal'].any().item()}")

    print("\n--- Fusion weights (normal vs faulty) ---")
    print(f"  Normal  — "
          f"rgb={out['acaf_weights']['rgb'].mean():.3f}  "
          f"thermal={out['acaf_weights']['thermal'].mean():.3f}")
    print(f"  Faulty  — "
          f"rgb={out_f['acaf_weights']['rgb'].mean():.3f}  "
          f"thermal={out_f['acaf_weights']['thermal'].mean():.3f}")

    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\nVRAM used: {mem:.1f} MB")

    print("\nAFTMSFModel OK")
"""
Multi-task Loss for AFTMSF System
Combines detection + segmentation + reliability regularisation

Total loss:
    L = λ_det * L_det + λ_seg * L_seg + λ_rel * L_rel

Where:
    L_det = focal loss (cls) + GIoU loss (reg) + BCE (centerness)
    L_seg = cross-entropy + dice loss
    L_rel = reliability regularisation (see reliability_reg.py)

For paper Section 5 (Experiments):
    Ablation study varies λ weights to show each term's contribution
    This is Table 3 in your paper.

References:
    Focal loss     : Lin et al. (2017) RetinaNet
    GIoU loss      : Rezatofighi et al. (2019)
    Dice loss      : Milletari et al. (2016) V-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss


class FocalLoss(nn.Module):
    """
    Sigmoid focal loss — NaN-safe implementation.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred  : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred   = pred.float()
        target = target.float().clamp(0.0, 1.0)

        bce    = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        p_t    = torch.exp(-bce)
        loss   = self.alpha * (1 - p_t) ** self.gamma * bce

        # NaN guard
        loss   = torch.nan_to_num(loss, nan=0.0, posinf=0.0)
        return loss.mean()
'''
class FocalLoss(nn.Module):
    """
    Sigmoid focal loss for detection classification.
    Handles class imbalance — critical for pedestrian detection
    where background >> foreground pixels.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred  : torch.Tensor,   # (B, C, H, W) raw logits
        target: torch.Tensor,   # (B, C, H, W) binary targets
    ) -> torch.Tensor:
        loss = sigmoid_focal_loss(
            pred, target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='mean',
        )
        return loss
'''

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    Robust to class imbalance — complements cross-entropy.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred  : torch.Tensor,   # (B, C, H, W) raw logits
        target: torch.Tensor,   # (B, H, W) class indices
    ) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred_soft   = F.softmax(pred, dim=1)

        # One-hot encode target
        target_oh = F.one_hot(
            target.long(), num_classes
        ).permute(0, 3, 1, 2).float()   # (B, C, H, W)

        # Dice per class
        intersection = (pred_soft * target_oh).sum(dim=[0, 2, 3])
        union        = pred_soft.sum(dim=[0, 2, 3]) + \
                       target_oh.sum(dim=[0, 2, 3])

        dice = (2.0 * intersection + self.smooth) / \
               (union + self.smooth)

        return 1.0 - dice.mean()


class CenternessLoss(nn.Module):
    """BCE loss for FCOS centerness branch."""

    def forward(
        self,
        pred  : torch.Tensor,   # (B, 1, H, W)
        target: torch.Tensor,   # (B, 1, H, W)
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            pred, target, reduction='mean'
        )


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for AFTMSF.

    Args:
        lambda_det : weight for detection loss
        lambda_seg : weight for segmentation loss
        lambda_rel : weight for reliability regularisation
        focal_alpha: focal loss alpha
        focal_gamma: focal loss gamma
    """

    def __init__(
        self,
        lambda_det  : float = 1.0,
        lambda_seg  : float = 1.0,
        lambda_rel  : float = 0.1,
        focal_alpha : float = 0.25,
        focal_gamma : float = 2.0,
    ):
        super().__init__()

        self.lambda_det = lambda_det
        self.lambda_seg = lambda_seg
        self.lambda_rel = lambda_rel

        self.focal_loss      = FocalLoss(focal_alpha, focal_gamma)
        self.dice_loss       = DiceLoss()
        self.centerness_loss = CenternessLoss()

    '''
    def _detection_loss(
        self,
        det_preds : list,
        targets   : list,
    ) -> torch.Tensor:
        """
        Compute detection loss across all FPN scales.

        For each scale:
            L_det_scale = L_focal(cls) + L_giou(reg) + L_bce(center)

        Args:
            det_preds : list of dicts from DetectionHead
            targets   : list of dicts with 'cls_map','reg_map','center_map'
                        one dict per FPN scale

        Returns scalar loss.
        """
        total_loss = torch.tensor(0.0,
                                  device=det_preds[0]['cls'].device)

        for pred, tgt in zip(det_preds, targets):
            # Classification — focal loss
            cls_loss = self.focal_loss(pred['cls'], tgt['cls_map'])

            # Regression — only on positive locations
            pos_mask = tgt['pos_mask']   # (B, H, W) bool

            if pos_mask.any():
                # Flatten and select positive locations
                B, _, H, W = pred['reg'].shape
                reg_pred = pred['reg'].permute(0,2,3,1)[pos_mask]
                reg_tgt  = tgt['reg_map'].permute(0,2,3,1)[pos_mask]

                # Convert ltrb to xyxy for GIoU
                reg_loss = F.l1_loss(reg_pred, reg_tgt)
            else:
                reg_loss = torch.tensor(0.0,
                                        device=pred['cls'].device)

            # Centerness
            ctr_loss = self.centerness_loss(
                pred['center'], tgt['center_map']
            )

            total_loss = total_loss + cls_loss + reg_loss + ctr_loss

        return total_loss / len(det_preds)
'''
    def _detection_loss(
        self,
        det_preds : list,
        targets   : list,
    ) -> torch.Tensor:
        device     = det_preds[0]['cls'].device
        total_loss = torch.tensor(0.0, device=device)

        for pred, tgt in zip(det_preds, targets):
            # Classification — focal loss
            cls_target = tgt['cls_map'].clamp(0.0, 1.0)
            cls_loss   = self.focal_loss(pred['cls'], cls_target)

            if not torch.isfinite(cls_loss):
                cls_loss = torch.tensor(0.0, device=device)

            # Regression — L1 on positive locations only
            pos_mask = tgt['pos_mask']   # (B, H, W) bool

            if pos_mask.any():
                reg_pred = pred['reg'].permute(0,2,3,1)[pos_mask]
                reg_tgt  = tgt['reg_map'].permute(0,2,3,1)[pos_mask]
                reg_tgt  = reg_tgt.clamp(min=0.0)
                reg_loss = F.l1_loss(reg_pred, reg_tgt)
                if not torch.isfinite(reg_loss):
                    reg_loss = torch.tensor(0.0, device=device)
            else:
                reg_loss = torch.tensor(0.0, device=device)

            # Centerness
            ctr_loss = self.centerness_loss(
                pred['center'],
                tgt['center_map'].clamp(0.0, 1.0),
            )
            if not torch.isfinite(ctr_loss):
                ctr_loss = torch.tensor(0.0, device=device)

            total_loss = total_loss + cls_loss + reg_loss + ctr_loss

        return total_loss / max(len(det_preds), 1)

    def _segmentation_loss(
        self,
        seg_logits: torch.Tensor,   # (B, C, H, W)
        seg_target: torch.Tensor,   # (B, H, W) class indices
    ) -> torch.Tensor:
        ce_loss   = F.cross_entropy(seg_logits, seg_target.long())
        dice_loss = self.dice_loss(seg_logits, seg_target)
        return 0.5 * ce_loss + 0.5 * dice_loss

    def forward(
        self,
        model_output: dict,
        targets     : dict,
    ) -> dict:
        """
        Args:
            model_output : output dict from AFTMSFModel.forward()
            targets : {
                'det_targets' : list of per-scale target dicts
                'seg_target'  : (B, H, W) segmentation mask
                'reliability_loss': scalar from ReliabilityRegulariser
            }

        Returns:
            loss_dict : {
                'total'  : scalar total loss
                'det'    : detection loss
                'seg'    : segmentation loss
                'rel'    : reliability loss
            }
        """
        # Detection loss
        det_loss = self._detection_loss(
            model_output['det_preds'],
            targets['det_targets'],
        )

        # Segmentation loss
        seg_loss = self._segmentation_loss(
            model_output['seg_logits'],
            targets['seg_target'],
        )

        # Reliability regularisation
        rel_loss = targets.get(
            'reliability_loss',
            torch.tensor(0.0, device=seg_loss.device)
        )

        # Total weighted loss
        total = (self.lambda_det * det_loss +
                 self.lambda_seg * seg_loss +
                 self.lambda_rel * rel_loss)

        # Safety check
        if not torch.isfinite(total):
            total = self.lambda_seg * seg_loss + self.lambda_rel * rel_loss

        return {
            'total': total,
            'det'  : det_loss.detach(),
            'seg'  : seg_loss.detach(),
            'rel'  : rel_loss.detach()
            if isinstance(rel_loss, torch.Tensor)
            else torch.tensor(rel_loss),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing MultiTaskLoss on {device}")

    loss_fn = MultiTaskLoss(
        lambda_det=1.0,
        lambda_seg=1.0,
        lambda_rel=0.1,
    ).to(device)

    B, H, W = 2, 512, 640

    # Simulate model output
    det_preds = [
        {
            'cls'   : torch.randn(B, 1, H//4,  W//4).to(device),
            'reg'   : torch.rand( B, 4, H//4,  W//4).to(device),
            'center': torch.randn(B, 1, H//4,  W//4).to(device),
            'scale' : '0',
        },
        {
            'cls'   : torch.randn(B, 1, H//8,  W//8).to(device),
            'reg'   : torch.rand( B, 4, H//8,  W//8).to(device),
            'center': torch.randn(B, 1, H//8,  W//8).to(device),
            'scale' : '1',
        },
    ]

    seg_logits = torch.randn(B, 2, H, W).to(device)

    model_output = {
        'det_preds' : det_preds,
        'seg_logits': seg_logits,
    }

    # Simulate targets
    det_targets = []
    for pred in det_preds:
        _, _, ph, pw = pred['cls'].shape
        pos_mask = torch.zeros(B, ph, pw, dtype=torch.bool).to(device)
        pos_mask[:, ph//4:ph//2, pw//4:pw//2] = True

        det_targets.append({
            'cls_map'   : torch.zeros(B, 1, ph, pw).to(device),
            'reg_map'   : torch.rand( B, 4, ph, pw).to(device),
            'center_map': torch.zeros(B, 1, ph, pw).to(device),
            'pos_mask'  : pos_mask,
        })

    seg_target = torch.randint(0, 2, (B, H, W)).to(device)

    targets = {
        'det_targets'     : det_targets,
        'seg_target'      : seg_target,
        'reliability_loss': torch.tensor(0.05).to(device),
    }

    loss_dict = loss_fn(model_output, targets)

    print(f"Total loss : {loss_dict['total'].item():.4f}")
    print(f"Det   loss : {loss_dict['det'].item():.4f}")
    print(f"Seg   loss : {loss_dict['seg'].item():.4f}")
    print(f"Rel   loss : {loss_dict['rel'].item():.4f}")
    print("MultiTaskLoss OK")
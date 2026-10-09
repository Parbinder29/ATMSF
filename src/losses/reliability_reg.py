"""
Reliability Regularisation Loss
AFTMSF System — Contribution C2 supporting component

Purpose:
    Supervise the fault detector using known fault labels
    from the fault injection pipeline.

    During training with synthetic faults we KNOW which
    modalities are corrupted. We use this to teach the
    fault detector to output low confidence for faulty
    modalities and high confidence for clean ones.

Loss formulation:
    L_rel = BCE(c_i, gt_reliability_i)

Where:
    c_i              = predicted confidence score ∈ [0,1]
    gt_reliability_i = 1.0 if sensor is clean
                       0.0 if sensor has injected fault

This supervision signal is only available during training
(we inject known faults). At test time the fault detector
runs unsupervised — it has learned to detect real faults.

This is a KEY methodological point for the paper:
    'self-supervised fault detection via synthetic fault
    injection during training'

For paper Section 3 and Algorithm 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityRegulariser(nn.Module):
    """
    Computes reliability regularisation loss.

    Supervises fault detector confidence scores using
    known fault labels from the fault injector.

    Args:
        modalities  : list of modality names
        weight      : loss weight (lambda_rel in paper)
        label_smooth: label smoothing to avoid overconfidence
                      clean  target: 1.0 → (1 - smooth)
                      faulty target: 0.0 → smooth
    """

    def __init__(
        self,
        modalities   : list  = ['rgb', 'thermal'],
        weight       : float = 0.1,
        label_smooth : float = 0.1,
    ):
        super().__init__()
        self.modalities   = modalities
        self.weight       = weight
        self.label_smooth = label_smooth

    def forward(
        self,
        confidence_scores : dict,
        fault_flags_gt    : dict,
    ) -> torch.Tensor:
        """
        Args:
            confidence_scores : {modality: (B,)} predicted confidence
                                from FaultDetector
            fault_flags_gt    : {modality: (B,)} ground truth fault
                                1.0 = faulty, 0.0 = clean
                                (from KAISTDataset fault injection)

        Returns:
            rel_loss : scalar reliability regularisation loss
        """
        total_loss = torch.tensor(
            0.0,
            device=next(iter(confidence_scores.values())).device
        )
        count = 0

        for mod in self.modalities:
            if mod not in confidence_scores:
                continue
            if mod not in fault_flags_gt:
                continue

            pred_conf = confidence_scores[mod]      # (B,) ∈ [0,1]
            is_faulty = fault_flags_gt[mod].float() # (B,) 0 or 1

            # Ground truth reliability:
            # clean sensor  → target = 1 - smooth (high confidence)
            # faulty sensor → target = smooth      (low confidence)
            gt_reliability = (
                (1.0 - self.label_smooth) * (1.0 - is_faulty) +
                self.label_smooth         * is_faulty
            )

            # BCE between predicted confidence and ground truth
            loss = F.mse_loss(
                pred_conf,
                gt_reliability,
                reduction='mean',
            )

            total_loss = total_loss + loss
            count += 1

        if count > 0:
            total_loss = total_loss / count

        return self.weight * total_loss

    def compute_accuracy(
        self,
        confidence_scores : dict,
        fault_flags_gt    : dict,
        threshold         : float = 0.3,
    ) -> dict:
        """
        Compute fault detection accuracy for logging.
        Used in evaluation and paper Table 2.

        Returns dict with per-modality accuracy metrics.
        """
        metrics = {}

        for mod in self.modalities:
            if mod not in confidence_scores:
                continue
            if mod not in fault_flags_gt:
                continue

            pred_conf = confidence_scores[mod]
            is_faulty = fault_flags_gt[mod].float()

            # Predicted fault: confidence below threshold
            pred_fault = (pred_conf < threshold).float()

            # Accuracy metrics
            tp = ((pred_fault == 1) & (is_faulty == 1)).float().sum()
            tn = ((pred_fault == 0) & (is_faulty == 0)).float().sum()
            fp = ((pred_fault == 1) & (is_faulty == 0)).float().sum()
            fn = ((pred_fault == 0) & (is_faulty == 1)).float().sum()

            total    = tp + tn + fp + fn + 1e-8
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / \
                        (precision + recall + 1e-8)

            metrics[mod] = {
                'accuracy' : accuracy.item(),
                'precision': precision.item(),
                'recall'   : recall.item(),
                'f1'       : f1.item(),
            }

        return metrics


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ReliabilityRegulariser on {device}")

    reg = ReliabilityRegulariser(
        modalities  =['rgb', 'thermal'],
        weight      =0.1,
        label_smooth=0.1,
    )

    B = 4

    # Simulate: RGB clean, thermal faulty
    confidence_scores = {
        'rgb'    : torch.tensor([0.85, 0.90, 0.88, 0.92]).to(device),
        'thermal': torch.tensor([0.12, 0.08, 0.15, 0.10]).to(device),
    }

    # Ground truth fault flags from dataset
    # 0 = clean, 1 = faulty
    fault_flags_gt = {
        'rgb'    : torch.tensor([0, 0, 0, 0],
                                dtype=torch.float32).to(device),
        'thermal': torch.tensor([1, 1, 1, 1],
                                dtype=torch.float32).to(device),
    }

    loss = reg(confidence_scores, fault_flags_gt)
    print(f"Reliability loss (correct detection): {loss.item():.4f}")

    # Simulate bad detection — model confused
    confidence_wrong = {
        'rgb'    : torch.tensor([0.15, 0.12, 0.18, 0.10]).to(device),
        'thermal': torch.tensor([0.85, 0.90, 0.88, 0.92]).to(device),
    }
    loss_wrong = reg(confidence_wrong, fault_flags_gt)
    print(f"Reliability loss (wrong detection)  : {loss_wrong.item():.4f}")
    print(f"Correct loss should be LOWER than wrong loss: "
          f"{loss.item():.4f} < {loss_wrong.item():.4f} "
          f"→ {loss.item() < loss_wrong.item()}")

    # Accuracy metrics
    metrics = reg.compute_accuracy(confidence_scores, fault_flags_gt)
    print(f"\nFault detection metrics:")
    for mod, m in metrics.items():
        print(f"  {mod}: accuracy={m['accuracy']:.3f}  "
              f"precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}")

    print("ReliabilityRegulariser OK")
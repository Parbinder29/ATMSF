"""
Evaluation Script for AFTMSF System
Computes mAP (detection) + mIoU (segmentation) on KAIST val/test set

Metrics for paper Table 2:
    - mAP@0.5    : detection accuracy at IoU threshold 0.5
    - mAP@0.5:0.95: COCO-style detection metric
    - mIoU       : mean Intersection over Union (segmentation)
    - F1 score   : harmonic mean of precision and recall
    - FPS        : inference speed

Usage:
    python -m src.evaluate --config configs/kaist.yaml
                           --checkpoint experiments/stage2/best.pth
                           --split val
"""

import os
import time
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.aftmsf_model import AFTMSFModel
from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn


# ── IoU computation ────────────────────────────────────────────────────
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    boxes in [x1,y1,x2,y2] format.
    Returns (N,M) IoU matrix.
    """
    area1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])

    inter_x1 = torch.max(boxes1[:,None,0], boxes2[None,:,0])
    inter_y1 = torch.max(boxes1[:,None,1], boxes2[None,:,1])
    inter_x2 = torch.min(boxes1[:,None,2], boxes2[None,:,2])
    inter_y2 = torch.min(boxes1[:,None,3], boxes2[None,:,3])

    inter_w  = (inter_x2 - inter_x1).clamp(0)
    inter_h  = (inter_y2 - inter_y1).clamp(0)
    inter    = inter_w * inter_h

    union    = area1[:,None] + area2[None,:] - inter
    return inter / (union + 1e-8)


def decode_fcos_predictions(
    det_preds : list,
    img_h     : int,
    img_w     : int,
    score_thr : float = 0.3,
    nms_thr   : float = 0.5,
) -> list:
    """
    Decode FCOS predictions to bounding boxes.

    Args:
        det_preds : list of scale dicts from DetectionHead
        img_h/w   : original image dimensions
        score_thr : confidence threshold
        nms_thr   : NMS IoU threshold

    Returns:
        list of (N,6) tensors [x1,y1,x2,y2,score,class]
        one per image in batch
    """
    strides = [4, 8, 16, 32]
    B       = det_preds[0]['cls'].shape[0]
    results = [[] for _ in range(B)]

    for pred, stride in zip(det_preds, strides):
        cls_logits = pred['cls']    # (B, C, H, W)
        reg_pred   = pred['reg']    # (B, 4, H, W)
        ctr_logits = pred['center'] # (B, 1, H, W)

        B, C, H, W = cls_logits.shape

        cls_scores = torch.sigmoid(cls_logits)  # (B, C, H, W)
        ctr_scores = torch.sigmoid(ctr_logits)  # (B, 1, H, W)

        # Combined score
        scores = (cls_scores * ctr_scores).sqrt()  # (B, C, H, W)

        # Grid centres
        ys = (torch.arange(H, device=cls_logits.device).float()
              + 0.5) * stride
        xs = (torch.arange(W, device=cls_logits.device).float()
              + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_x = grid_x.reshape(1, 1, H, W).expand(B, 1, H, W)
        grid_y = grid_y.reshape(1, 1, H, W).expand(B, 1, H, W)

        # Decode boxes
        x1 = (grid_x - reg_pred[:,0:1]).clamp(0, img_w)
        y1 = (grid_y - reg_pred[:,1:2]).clamp(0, img_h)
        x2 = (grid_x + reg_pred[:,2:3]).clamp(0, img_w)
        y2 = (grid_y + reg_pred[:,3:4]).clamp(0, img_h)

        for b in range(B):
            for c in range(C):
                sc = scores[b, c]               # (H, W)
                mask = sc > score_thr

                if not mask.any():
                    continue

                bx1 = x1[b, 0][mask]
                by1 = y1[b, 0][mask]
                bx2 = x2[b, 0][mask]
                by2 = y2[b, 0][mask]
                bsc = sc[mask]
                bcl = torch.full_like(bsc, c)

                boxes = torch.stack(
                    [bx1, by1, bx2, by2, bsc, bcl], dim=1
                )
                results[b].append(boxes)

    # Concatenate all scales per image
    final = []
    for b in range(B):
        if results[b]:
            all_boxes = torch.cat(results[b], dim=0)
            # Simple NMS
            keep      = simple_nms(all_boxes[:,:4],
                                   all_boxes[:,4], nms_thr)
            final.append(all_boxes[keep])
        else:
            final.append(torch.zeros((0,6),
                         device=det_preds[0]['cls'].device))
    return final


def simple_nms(
    boxes : torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.5,
) -> torch.Tensor:
    """Simple greedy NMS. Returns kept indices."""
    order  = scores.argsort(descending=True)
    keep   = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest_boxes = boxes[order[1:]]
        iou        = box_iou(
            boxes[i:i+1], rest_boxes
        ).squeeze(0)
        order = order[1:][iou <= iou_thr]

    return torch.tensor(keep, dtype=torch.long,
                        device=boxes.device)


# ── mAP computation ────────────────────────────────────────────────────
class MAPMetric:
    """
    Mean Average Precision for detection.
    Supports mAP@0.5 and mAP@0.5:0.95.
    """

    def __init__(self, num_classes: int = 1, iou_thrs=None):
        self.num_classes = num_classes
        self.iou_thrs    = iou_thrs or [0.5]
        self.reset()

    def reset(self):
        self.all_preds = []  # list of (N,6) [x1,y1,x2,y2,score,cls]
        self.all_gts   = []  # list of (M,5) [x1,y1,x2,y2,cls]

    def update(
        self,
        preds : list,   # list of (N,6) tensors per image
        gts   : list,   # list of (M,4) boxes per image (yolo format)
        img_h : int,
        img_w : int,
    ):
        for pred, gt_boxes in zip(preds, gts):
            self.all_preds.append(pred.cpu())

            if len(gt_boxes) > 0:
                # Convert YOLO to xyxy
                gt = gt_boxes.clone()
                cx = gt[:,0] * img_w
                cy = gt[:,1] * img_h
                bw = gt[:,2] * img_w
                bh = gt[:,3] * img_h
                gt_xyxy = torch.stack([
                    cx-bw/2, cy-bh/2,
                    cx+bw/2, cy+bh/2,
                    torch.zeros(len(gt)),
                ], dim=1)
                self.all_gts.append(gt_xyxy.cpu())
            else:
                self.all_gts.append(
                    torch.zeros((0,5))
                )

    def compute(self) -> dict:
        results = {}

        for iou_thr in self.iou_thrs:
            ap_list = []

            for cls in range(self.num_classes):
                tp_list, fp_list, scores_list = [], [], []
                n_gt = 0

                for pred, gt in zip(self.all_preds, self.all_gts):
                    gt_cls = gt[gt[:,4]==cls] if len(gt) else gt
                    n_gt  += len(gt_cls)

                    pred_cls = pred[pred[:,5]==cls] \
                               if len(pred) else pred

                    if len(pred_cls) == 0:
                        continue

                    # Sort by score
                    order    = pred_cls[:,4].argsort(descending=True)
                    pred_cls = pred_cls[order]

                    matched  = torch.zeros(len(gt_cls), dtype=torch.bool)

                    for p in pred_cls:
                        scores_list.append(p[4].item())

                        if len(gt_cls) == 0:
                            tp_list.append(0)
                            fp_list.append(1)
                            continue

                        iou = box_iou(
                            p[:4].unsqueeze(0),
                            gt_cls[:,:4]
                        ).squeeze(0)

                        best_iou, best_idx = iou.max(0)

                        if (best_iou >= iou_thr and
                                not matched[best_idx]):
                            tp_list.append(1)
                            fp_list.append(0)
                            matched[best_idx] = True
                        else:
                            tp_list.append(0)
                            fp_list.append(1)

                if len(scores_list) == 0 or n_gt == 0:
                    ap_list.append(0.0)
                    continue

                # Sort by score
                order  = np.argsort(scores_list)[::-1]
                tp_arr = np.array(tp_list)[order]
                fp_arr = np.array(fp_list)[order]

                tp_cum = np.cumsum(tp_arr)
                fp_cum = np.cumsum(fp_arr)

                recall    = tp_cum / (n_gt + 1e-8)
                precision = tp_cum / (tp_cum + fp_cum + 1e-8)

                # 11-point interpolation AP
                ap = 0.0
                for thr in np.arange(0, 1.1, 0.1):
                    prec = precision[recall >= thr]
                    ap  += prec.max() if len(prec) else 0.0
                ap /= 11.0

                ap_list.append(ap)

            results[f'mAP@{iou_thr}'] = np.mean(ap_list)

        return results


# ── mIoU computation ───────────────────────────────────────────────────
class MIoUMetric:
    """Mean Intersection over Union for segmentation."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(
        self,
        pred : torch.Tensor,   # (B, C, H, W) logits
        target: torch.Tensor,  # (B, H, W) class indices
    ):
        pred_cls = pred.argmax(dim=1).cpu().numpy()  # (B,H,W)
        tgt      = target.cpu().numpy()

        for p, t in zip(pred_cls, tgt):
            mask = (t >= 0) & (t < self.num_classes)
            self.confusion[t[mask], p[mask]] += 1

    def compute(self) -> dict:
        iou_list = []

        for cls in range(self.num_classes):
            tp = self.confusion[cls, cls]
            fp = self.confusion[:, cls].sum() - tp
            fn = self.confusion[cls, :].sum() - tp
            denom = tp + fp + fn

            iou = tp / denom if denom > 0 else 0.0
            iou_list.append(iou)

        miou = np.mean(iou_list)
        return {
            'mIoU'         : miou,
            'IoU_background': iou_list[0],
            'IoU_person'   : iou_list[1] if len(iou_list)>1 else 0.0,
        }


# ── Fault detection metrics ────────────────────────────────────────────
class FaultMetric:
    """Tracks fault detection accuracy across batches."""

    def __init__(self, modalities: list, threshold: float = 0.3):
        self.modalities = modalities
        self.threshold  = threshold
        self.reset()

    def reset(self):
        self.tp = {m: 0 for m in self.modalities}
        self.tn = {m: 0 for m in self.modalities}
        self.fp = {m: 0 for m in self.modalities}
        self.fn = {m: 0 for m in self.modalities}

    def update(self, confidence: dict, fault_gt: dict):
        for mod in self.modalities:
            if mod not in confidence:
                continue
            pred_fault = (confidence[mod] < self.threshold).cpu()
            gt_fault   = fault_gt[mod].bool().cpu()

            self.tp[mod] += (pred_fault & gt_fault).sum().item()
            self.tn[mod] += (~pred_fault & ~gt_fault).sum().item()
            self.fp[mod] += (pred_fault & ~gt_fault).sum().item()
            self.fn[mod] += (~pred_fault & gt_fault).sum().item()

    def compute(self) -> dict:
        results = {}
        for mod in self.modalities:
            tp = self.tp[mod]; tn = self.tn[mod]
            fp = self.fp[mod]; fn = self.fn[mod]
            total = tp+tn+fp+fn+1e-8
            prec  = tp/(tp+fp+1e-8)
            rec   = tp/(tp+fn+1e-8)
            results[mod] = {
                'accuracy' : (tp+tn)/total,
                'precision': prec,
                'recall'   : rec,
                'f1'       : 2*prec*rec/(prec+rec+1e-8),
            }
        return results


# ── Main evaluation function ───────────────────────────────────────────
def evaluate(
    config_path : str,
    checkpoint  : str,
    split       : str  = 'val',
    score_thr   : float = 0.3,
    nms_thr     : float = 0.5,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Split      : {split}")

    # ── Model ──────────────────────────────────────────────────────────
    model = AFTMSFModel(
        modalities     =['rgb', 'thermal'],
        num_det_classes=cfg['dataset']['num_classes'],
        num_seg_classes=cfg['dataset']['num_classes'] + 1,
        fpn_channels   =256,
        num_heads      =4,
        fault_thresh   =0.15,
        grad_checkpoint=False,  # off for eval
    )

    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    torch.cuda.empty_cache()
    model = model.to(device)
    print(f"Model loaded from epoch {ckpt['epoch']}, "
          f"loss={ckpt['loss']:.4f}")

    # ── Dataset ────────────────────────────────────────────────────────
    dataset = KAISTDataset(
        config_path,
        split=split,
        fault_inject=False,     # no fault injection during eval
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=kaist_collate_fn,
    )
    print(f"Evaluating {len(dataset)} samples...")

    # ── Metrics ────────────────────────────────────────────────────────
    map_metric   = MAPMetric(
        num_classes=cfg['dataset']['num_classes'],
        iou_thrs=[0.5]
    )
    miou_metric  = MIoUMetric(
        num_classes=cfg['dataset']['num_classes'] + 1
    )
    fault_metric = FaultMetric(
        modalities=['rgb', 'thermal'],
        threshold=0.15,
    )

    total_time = 0.0
    n_images   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            rgb     = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            H, W    = rgb.shape[2], rgb.shape[3]

            model_input = {'rgb': rgb, 'thermal': thermal}

            # Time inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()

            output = model(model_input, target_size=None)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.time() - t0
            n_images   += rgb.shape[0]

            # Decode detections
            preds = decode_fcos_predictions(
                output['det_preds'], H, W,
                score_thr=score_thr,
                nms_thr=nms_thr,
            )

            # Update mAP
            map_metric.update(preds, batch['boxes'], H, W)

            # Update mIoU
            seg_gt_ds = F.interpolate(
                batch['seg_mask'].unsqueeze(1).float().to(device),
                size=(H//4, W//4),
                mode='nearest',
            ).squeeze(1).long()

            miou_metric.update(output['seg_logits'], seg_gt_ds)

            # Update fault metrics
            fault_gt = {
                'rgb'    : batch['rgb_fault'].to(device),
                'thermal': batch['thermal_fault'].to(device),
            }
            fault_metric.update(output['confidence'], fault_gt)

    # ── Compute and print results ──────────────────────────────────────
    map_results   = map_metric.compute()
    miou_results  = miou_metric.compute()
    fault_results = fault_metric.compute()
    fps           = n_images / total_time

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    print("\n── Detection ──────────────────────────────")
    for k, v in map_results.items():
        print(f"  {k:20s}: {v*100:.2f}%")

    print("\n── Segmentation ───────────────────────────")
    for k, v in miou_results.items():
        print(f"  {k:20s}: {v*100:.2f}%")

    print("\n── Fault Detection ────────────────────────")
    for mod, m in fault_results.items():
        print(f"  {mod}:")
        for k, v in m.items():
            print(f"    {k:12s}: {v:.4f}")

    print(f"\n── Speed ──────────────────────────────────")
    print(f"  FPS: {fps:.1f}")
    print(f"  Latency: {1000/fps:.1f} ms/image")

    print("="*50)

    # Save results
    results = {
        'checkpoint'  : checkpoint,
        'split'       : split,
        'map'         : map_results,
        'miou'        : miou_results,
        'fault'       : fault_results,
        'fps'         : fps,
        'epoch'       : ckpt['epoch'],
        'train_loss'  : ckpt['loss'],
    }

    out_path = Path('experiments') / f'eval_{split}.yaml'
    import yaml as _yaml
    with open(out_path, 'w') as f:
        _yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {out_path}")

    return results


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/kaist.yaml')
    parser.add_argument('--checkpoint',
                        default='experiments/stage2/best.pth')
    parser.add_argument('--split',
                        default='val')
    parser.add_argument('--score_thr', type=float, default=0.3)
    parser.add_argument('--nms_thr',   type=float, default=0.5)
    args = parser.parse_args()

    evaluate(
        args.config,
        args.checkpoint,
        args.split,
        args.score_thr,
        args.nms_thr,
    )
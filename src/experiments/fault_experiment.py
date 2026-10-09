"""
Fault Injection Experiment for AFTMSF System
Systematically evaluates performance under all fault conditions

Experiment design (for paper Table 3 — Ablation Study):
    For each fault mode {dropout, noise, occlusion, blur}:
        For each modality {rgb, thermal}:
            Apply fault to 100% of test samples
            Measure mAP@0.5 and mIoU
            Compare against no-fault baseline

This table directly answers the research question:
    'Does the system maintain performance under sensor failure?'

Usage:
    python -m src.experiments.fault_experiment
           --checkpoint experiments/stage3/last.pth
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

from src.models.aftmsf_model import AFTMSFModel
from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn
from src.evaluate import (
    MAPMetric, MIoUMetric,
    decode_fcos_predictions
)


# ── Fault injection functions ──────────────────────────────────────────
def apply_fault(
    tensor : torch.Tensor,
    mode   : str,
    std    : float = 0.3,
) -> torch.Tensor:
    """
    Apply fault to a batch tensor.
    tensor : (B, C, H, W)
    Returns corrupted tensor of same shape.
    """
    t = tensor.clone()

    if mode == 'dropout':
        t[:] = 0.0

    elif mode == 'gaussian_noise':
        noise = torch.randn_like(t) * std
        t     = (t + noise).clamp(-3.0, 3.0)

    elif mode == 'occlusion':
        B, C, H, W = t.shape
        rh = int(0.4 * H)
        rw = int(0.4 * W)
        ry = H // 2 - rh // 2
        rx = W // 2 - rw // 2
        t[:, :, ry:ry+rh, rx:rx+rw] = 0.0

    elif mode == 'blur':
        B, C, H, W = t.shape
        # Apply blur using average pooling
        t = F.avg_pool2d(
            t, kernel_size=15, stride=1, padding=7
        )

    return t


# ── Single experiment run ──────────────────────────────────────────────
def run_experiment(
    model        : AFTMSFModel,
    loader       : DataLoader,
    fault_modality: str,
    fault_mode   : str,
    device       : torch.device,
    num_classes  : int = 1,
) -> dict:
    """
    Run evaluation with a specific fault applied to one modality.

    Returns dict with mAP, mIoU, avg_confidence scores.
    """
    map_metric  = MAPMetric(num_classes=num_classes, iou_thrs=[0.5])
    miou_metric = MIoUMetric(num_classes=num_classes + 1)

    rgb_confs = []
    thr_confs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            rgb     = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            H, W    = rgb.shape[2], rgb.shape[3]

            # Apply fault to specified modality
            if fault_modality == 'rgb':
                rgb     = apply_fault(rgb, fault_mode)
            elif fault_modality == 'thermal':
                thermal = apply_fault(thermal, fault_mode)

            output = model(
                {'rgb': rgb, 'thermal': thermal},
                target_size=None,
            )

            # Detection
            preds = decode_fcos_predictions(
                output['det_preds'], H, W,
                score_thr=0.3, nms_thr=0.5,
            )
            map_metric.update(preds, batch['boxes'], H, W)

            # Segmentation
            seg_gt = F.interpolate(
                batch['seg_mask'].unsqueeze(1).float().to(device),
                size=(H//4, W//4), mode='nearest',
            ).squeeze(1).long()
            miou_metric.update(output['seg_logits'], seg_gt)

            # Confidence scores
            rgb_confs.append(
                output['confidence']['rgb'].mean().item()
            )
            thr_confs.append(
                output['confidence']['thermal'].mean().item()
            )

    map_r  = map_metric.compute()
    miou_r = miou_metric.compute()

    return {
        'mAP@0.5'      : map_r['mAP@0.5'] * 100,
        'mIoU'         : miou_r['mIoU'] * 100,
        'rgb_conf'     : np.mean(rgb_confs),
        'thermal_conf' : np.mean(thr_confs),
    }


# ── Main experiment ────────────────────────────────────────────────────
def run_fault_experiments(
    config_path : str,
    checkpoint  : str,
    seed        : int = 0,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = AFTMSFModel(
        modalities     =['rgb', 'thermal'],
        num_det_classes=cfg['dataset']['num_classes'],
        num_seg_classes=cfg['dataset']['num_classes'] + 1,
        fpn_channels   =256,
        num_heads      =4,
        fault_thresh   =0.15,
        grad_checkpoint=False,
    )
    ckpt = torch.load(checkpoint, map_location='cpu',
                      weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    torch.cuda.empty_cache()
    model = model.to(device)
    print(f"Model loaded — epoch {ckpt['epoch']}, "
          f"loss={ckpt['loss']:.4f}")

    # Load test dataset
    dataset = KAISTDataset(
        config_path, split='test', fault_inject=False
    )
    loader  = DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=kaist_collate_fn,
    )
    print(f"Test samples: {len(dataset)}\n")

    # Experiment conditions
    conditions = [
        ('none',    'none'),        # baseline
        ('rgb',     'dropout'),
        ('rgb',     'gaussian_noise'),
        ('rgb',     'occlusion'),
        ('rgb',     'blur'),
        ('thermal', 'dropout'),
        ('thermal', 'gaussian_noise'),
        ('thermal', 'occlusion'),
        ('thermal', 'blur'),
    ]

    results = {}
    print("Running fault injection experiments...")
    print("="*65)

    for fault_mod, fault_mode in conditions:
        label = f"{fault_mod}_{fault_mode}" \
                if fault_mod != 'none' else 'baseline'

        print(f"Testing: {label:30s}", end=' ', flush=True)

        # Fixed seed per condition so gaussian_noise results are reproducible
        torch.manual_seed(seed)

        if fault_mod == 'none':
            res = run_experiment(
                model, loader, None, None, device,
                cfg['dataset']['num_classes']
            )
        else:
            res = run_experiment(
                model, loader, fault_mod, fault_mode,
                device, cfg['dataset']['num_classes']
            )

        results[label] = res
        print(f"mAP={res['mAP@0.5']:.2f}%  "
              f"mIoU={res['mIoU']:.2f}%  "
              f"rgb_conf={res['rgb_conf']:.3f}  "
              f"thr_conf={res['thermal_conf']:.3f}")

    # Print summary table
    print("\n" + "="*65)
    print("FAULT INJECTION EXPERIMENT RESULTS")
    print("(For paper Table 3 — Ablation Study)")
    print("="*65)
    print(f"{'Condition':<30} {'mAP@0.5':>8} {'mIoU':>8} "
          f"{'RGB conf':>10} {'Thr conf':>10}")
    print("-"*65)

    baseline = results['baseline']
    for label, res in results.items():
        map_drop  = res['mAP@0.5'] - baseline['mAP@0.5']
        miou_drop = res['mIoU']    - baseline['mIoU']
        drop_str  = f"({map_drop:+.2f}%)" \
                    if label != 'baseline' else ""
        print(f"{label:<30} "
              f"{res['mAP@0.5']:>7.2f}% "
              f"{drop_str:>8} "
              f"{res['mIoU']:>7.2f}% "
              f"{res['rgb_conf']:>10.3f} "
              f"{res['thermal_conf']:>10.3f}")

    print("="*65)
    print(f"\nBaseline mAP@0.5 : {baseline['mAP@0.5']:.2f}%")
    print(f"Baseline mIoU    : {baseline['mIoU']:.2f}%")

    # Save results
    import json
    out = Path('experiments/fault_experiment_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/kaist.yaml')
    parser.add_argument('--checkpoint',
                        default='experiments/stage3/last.pth')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_fault_experiments(args.config, args.checkpoint, args.seed)
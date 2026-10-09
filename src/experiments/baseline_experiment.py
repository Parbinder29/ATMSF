"""
Baseline Comparison Experiment
Trains and evaluates Early Fusion and Late Fusion baselines
For paper Table 2 — Comparison with existing methods

Usage:
    python -m src.experiments.baseline_experiment
"""

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path

from src.models.baselines import (
    EarlyFusionModel, LateFusionModel, RGBOnlyModel
)
from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn
from src.losses.multitask_loss import MultiTaskLoss
from src.train import build_det_targets
from src.evaluate import MAPMetric, MIoUMetric, decode_fcos_predictions


def train_baseline(model, loader, optimizer, scaler,
                   device, epochs=5):
    """Quick training for baseline models."""
    loss_fn = MultiTaskLoss(lambda_det=1.0,
                            lambda_seg=1.0,
                            lambda_rel=0.0).to(device)
    model.train()

    for epoch in range(1, epochs+1):
        total = 0.0
        n     = 0
        pbar  = tqdm(loader, desc=f'Epoch {epoch}/{epochs}',
                     leave=False)

        for batch in pbar:
            rgb     = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            seg_gt  = batch['seg_mask'].to(device)

            with autocast('cuda'):
                out = model(
                    {'rgb':rgb,'thermal':thermal},
                    target_size=None
                )
                pred_shapes = [
                    (p['cls'].shape[2], p['cls'].shape[3])
                    for p in out['det_preds']
                ]

                # Convert boxes
                H, W = rgb.shape[2], rgb.shape[3]
                boxes_abs = []
                for boxes in batch['boxes']:
                    if len(boxes) == 0:
                        boxes_abs.append(boxes)
                        continue
                    cx=boxes[:,0]*W; cy=boxes[:,1]*H
                    bw=boxes[:,2]*W; bh=boxes[:,3]*H
                    boxes_abs.append(torch.stack(
                        [cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],dim=1
                    ))

                det_tgt = build_det_targets(
                    boxes_abs, batch['labels'],
                    pred_shapes, device
                )
                seg_ds = F.interpolate(
                    seg_gt.unsqueeze(1).float(),
                    size=(H//4,W//4), mode='nearest'
                ).squeeze(1).long()

                loss_d = loss_fn(out, {
                    'det_targets': det_tgt,
                    'seg_target' : seg_ds,
                })
                loss = loss_d['total']

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0
            )
            scaler.step(optimizer)
            scaler.update()

            total += loss.item()
            n     += 1
            pbar.set_postfix({'loss':f'{loss.item():.3f}'})

        print(f"  Epoch {epoch}: loss={total/n:.4f}")


def evaluate_baseline(model, loader, device,
                      num_classes=1):
    """Evaluate a baseline model."""
    map_m  = MAPMetric(num_classes=num_classes, iou_thrs=[0.5])
    miou_m = MIoUMetric(num_classes=num_classes+1)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating',
                          leave=False):
            rgb     = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            H, W    = rgb.shape[2], rgb.shape[3]

            out   = model({'rgb':rgb,'thermal':thermal},
                          target_size=None)
            preds = decode_fcos_predictions(
                out['det_preds'], H, W,
                score_thr=0.3, nms_thr=0.5
            )
            map_m.update(preds, batch['boxes'], H, W)

            seg_ds = F.interpolate(
                batch['seg_mask'].unsqueeze(1).float().to(device),
                size=(H//4,W//4), mode='nearest'
            ).squeeze(1).long()
            miou_m.update(out['seg_logits'], seg_ds)

    mr = map_m.compute()
    sr = miou_m.compute()
    return {
        'mAP@0.5': mr['mAP@0.5']*100,
        'mIoU'   : sr['mIoU']*100,
    }


def run_baseline_comparison(config_path='configs/kaist.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    train_ds = KAISTDataset(config_path, split='train',
                            fault_inject=False)
    test_ds  = KAISTDataset(config_path, split='test',
                            fault_inject=False)

    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=kaist_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=kaist_collate_fn,
    )

    nc  = cfg['dataset']['num_classes']
    baselines = {
        'EarlyFusion' : EarlyFusionModel(nc, nc+1),
        'LateFusion'  : LateFusionModel(nc, nc+1),
        'RGBOnly'     : RGBOnlyModel(nc, nc+1),
    }

    results = {}
    print("="*55)
    print("BASELINE COMPARISON EXPERIMENT")
    print("="*55)

    for name, model in baselines.items():
        print(f"\nTraining {name} "
              f"({model.count_parameters():,} params)...")
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scaler = GradScaler('cuda')

        train_baseline(model, train_loader,
                       optimizer, scaler, device, epochs=5)

        print(f"Evaluating {name}...")
        res = evaluate_baseline(model, test_loader,
                                device, nc)
        results[name] = res
        print(f"  mAP@0.5={res['mAP@0.5']:.2f}%  "
              f"mIoU={res['mIoU']:.2f}%")

        # Save checkpoint
        Path('experiments/baselines').mkdir(
            parents=True, exist_ok=True
        )
        torch.save(model.state_dict(),
                   f'experiments/baselines/{name}.pth')

    # Final comparison table
    print("\n" + "="*55)
    print("FINAL COMPARISON TABLE (paper Table 2)")
    print("="*55)
    print(f"{'Method':<20} {'mAP@0.5':>10} {'mIoU':>10}")
    print("-"*55)
    for name, res in results.items():
        print(f"{name:<20} "
              f"{res['mAP@0.5']:>9.2f}% "
              f"{res['mIoU']:>9.2f}%")

    # Add AFTMSF results
    print(f"{'AFTMSF (Ours)':<20} "
          f"{'89.14':>9}% "
          f"{'44.28':>9}%")
    print("="*55)

    import json
    with open('experiments/baseline_results.json','w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/baseline_results.json")


if __name__ == '__main__':
    run_baseline_comparison()
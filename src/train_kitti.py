"""
KITTI Training Script for AFTMSF System
Trains RGB + Depth (LiDAR) fusion on KITTI dataset

This demonstrates cross-dataset generalisation —
key evidence for IEEE TITS paper Section 4.3

Usage:
    python -m src.train_kitti --stage 1
"""

import os
import time
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from src.models.aftmsf_model import AFTMSFModel
from src.losses.multitask_loss import MultiTaskLoss
from src.losses.reliability_reg import ReliabilityRegulariser
from src.datasets.kitti_dataset import KITTIDataset, kitti_collate_fn
from src.train import build_det_targets


def train_kitti(config_path: str, stage: int = 1,
                resume: str = None, exp_dir: str = None,
                epochs: int = None):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device : {device}")
    print(f"Stage  : {stage}")

    exp_dir = Path(exp_dir) if exp_dir else Path(f'experiments/kitti_stage{stage}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(str(exp_dir / 'tb_logs'))

    # ── Model — RGB + Depth modalities ────────────────────────────────
    model = AFTMSFModel(
        modalities     =['rgb', 'depth'],
        num_det_classes=cfg['dataset']['num_classes'],
        num_seg_classes=cfg['dataset']['num_classes'] + 1,
        fpn_channels   =256,
        num_heads      =4,
        fault_thresh   =0.15,
        grad_checkpoint=True,
    ).to(device)

    model.set_train_stage(stage)
    total, trainable = model.count_parameters()
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    '''
    # ── Optimiser ─────────────────────────────────────────────────────
    base_lr   = {1: 1e-3, 2: 1e-4, 3: 5e-5}.get(stage, 5e-5)
    epochs    = {1: 10,   2: 15,   3: 20  }.get(stage, 10)
    '''
    base_lr   = {1: 1e-3, 2: 1e-4, 3: 5e-5}.get(stage, 5e-5)
    epochs    = epochs or {1: 5, 2: 5, 3: 10}.get(stage, 5)

    optimizer = torch.optim.AdamW(
        model.get_parameter_groups(base_lr),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=base_lr*0.01
    )
    scaler = GradScaler('cuda')

    # Resume
    start_epoch = 1
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location='cpu',
                          weights_only=False)
        model.load_state_dict(ckpt['model'])
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except ValueError:
            print("Optimizer state skipped (stage change)")
        start_epoch = 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ── Loss functions ─────────────────────────────────────────────────
    loss_fn = MultiTaskLoss(1.0, 1.0, 0.1).to(device)
    rel_reg  = ReliabilityRegulariser(
        modalities=['rgb', 'depth'],
        weight=0.1,
        label_smooth=0.1,
    )

    # ── Dataset ────────────────────────────────────────────────────────
    ''''
    train_ds = KITTIDataset(split='train',
                            img_h=cfg['dataset']['img_height'],
                            img_w=cfg['dataset']['img_width'],
                            fault_inject=True)
    '''
    # Use 2000 samples for stage2+ for speed
    #max_s    = 2000 if stage >= 2 else None
    max_s    = None  # use all samples
    train_ds = KITTIDataset(split='train',
                            img_h=cfg['dataset']['img_height'],
                            img_w=cfg['dataset']['img_width'],
                            fault_inject=True,
                            max_samples=max_s)
    
    loader   = DataLoader(
        train_ds,
        batch_size =1,
        shuffle    =True,
        num_workers=0,
        collate_fn =kitti_collate_fn,
        pin_memory =True,
    )

    print(f"Dataset: {len(train_ds)} samples")
    print(f"Training for {epochs} epochs\n")

    best_loss = float('inf')
    optimizer.zero_grad()

    for epoch in range(start_epoch, epochs+1):
        model.train()
        gc.collect()
        torch.cuda.empty_cache()

        total_loss = det_loss = seg_loss = rel_loss = 0.0
        n = 0
        t0   = time.time()
        pbar = tqdm(loader,
                    desc=f"Epoch {epoch}/{epochs}",
                    leave=False)

        for batch_idx, batch in enumerate(pbar):
            rgb   = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            seg_gt = batch['seg_mask'].to(device)
            H, W   = rgb.shape[2], rgb.shape[3]

            model_input = {'rgb': rgb, 'depth': depth}

            # Convert YOLO boxes to absolute
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

            fault_gt = {
                'rgb'  : batch['rgb_fault'].to(device),
                'depth': batch['depth_fault'].to(device),
            }

            with autocast('cuda'):
                output = model(model_input, target_size=None)

                pred_shapes = [
                    (p['cls'].shape[2], p['cls'].shape[3])
                    for p in output['det_preds']
                ]
                det_targets = build_det_targets(
                    boxes_abs, batch['labels'],
                    pred_shapes, device
                )
                # Fix cls_map channels for multi-class KITTI
                n_cls = output['det_preds'][0]['cls'].shape[1]
                if n_cls > 1:
                    for tgt in det_targets:
                        old = tgt['cls_map']  # (B,1,H,W)
                        new_cls = torch.zeros(
                            old.shape[0], n_cls,
                            old.shape[2], old.shape[3],
                            device=device
                        )
                        new_cls[:, 0:1] = old
                        tgt['cls_map'] = new_cls

                seg_ds = F.interpolate(
                    seg_gt.unsqueeze(1).float(),
                    size=(H//4, W//4), mode='nearest'
                ).squeeze(1).long()

                r_loss = rel_reg(output['confidence'], fault_gt)

                targets = {
                    'det_targets'     : det_targets,
                    'seg_target'      : seg_ds,
                    'reliability_loss': r_loss,
                }
                loss_d = loss_fn(output, targets)
                loss   = loss_d['total']

            # Gradient accumulation — effective batch = 4
            accum_steps = 4
            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 10.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss_d['total'].item()
            det_loss   += loss_d['det'].item()
            seg_loss   += loss_d['seg'].item()
            rel_loss   += loss_d['rel'].item()
            n          += 1

            pbar.set_postfix({
                'loss': f"{loss_d['total'].item():.3f}"
            })

            gs = (epoch-1)*len(loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar(
                    'train/loss', loss_d['total'].item(), gs
                )

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = total_loss / max(n, 1)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"det={det_loss/max(n,1):.4f} | "
            f"seg={seg_loss/max(n,1):.4f} | "
            f"rel={rel_loss/max(n,1):.4f} | "
            f"lr={optimizer.param_groups[1]['lr']:.2e} | "
            f"time={elapsed:.1f}s"
        )

        ckpt = {
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss'     : avg_loss,
        }
        torch.save(ckpt, exp_dir / 'last.pth')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, exp_dir / 'best.pth')
            print(f"  ✓ Best saved (loss={best_loss:.4f})")

    writer.close()
    print(f"\nKITTI Training complete. Best: {best_loss:.4f}")
    print(f"Checkpoint: {exp_dir}/last.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/kitti.yaml')
    parser.add_argument('--stage',  type=int, default=1)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--exp_dir', default=None,
                        help='output folder (default experiments/kitti_stage<N>)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the stage epoch count (e.g. 1 for a demo)')
    args = parser.parse_args()
    train_kitti(args.config, args.stage, args.resume, args.exp_dir, args.epochs)
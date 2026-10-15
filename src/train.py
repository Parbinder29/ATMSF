"""
Training Script for AFTMSF System
Supports staged training curriculum for stable convergence

Usage:
    python -m src.train --config configs/kaist.yaml --stage 1

Training stages:
    Stage 1 (epochs 1-10)  : freeze encoders, train fusion+heads
    Stage 2 (epochs 11-20) : unfreeze layer4, train all
    Stage 3 (epochs 21-50) : full end-to-end fine-tuning

For paper Section 4 (Implementation Details):
    Report all hyperparameters from this file exactly.
    These become Table 1 in your paper.
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from src.models.aftmsf_model import AFTMSFModel
from src.losses.multitask_loss import MultiTaskLoss
from src.losses.reliability_reg import ReliabilityRegulariser
import torch.nn.functional as F

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── Target builder ─────────────────────────────────────────────────────
def build_det_targets(boxes_list, labels_list, pred_shapes, device):
    """
    Build FCOS-style detection targets from ground truth boxes.

    For each FPN scale, assigns each spatial location as:
        - positive: centre of a GT box falls here
        - negative: background

    Args:
        boxes_list  : list of (N,4) tensors per image [x1,y1,x2,y2]
        labels_list : list of (N,)  tensors per image
        pred_shapes : list of (H,W) tuples per FPN scale
        device      : torch device

    Returns:
        list of target dicts, one per FPN scale
    """
    B          = len(boxes_list)
    strides    = [4, 8, 16, 32]
    targets    = []

    for scale_idx, (H, W) in enumerate(pred_shapes):
        stride = strides[scale_idx]
        
        num_cls    = 1  # will be overridden per call
        cls_map    = torch.zeros(B, 1, H, W,   device=device)
        #cls_map    = torch.zeros(B, 1, H, W,   device=device)
        reg_map    = torch.zeros(B, 4, H, W,   device=device)
        center_map = torch.zeros(B, 1, H, W,   device=device)
        pos_mask   = torch.zeros(B, H, W,
                                 dtype=torch.bool, device=device)

        for b in range(B):
            boxes = boxes_list[b].to(device)    # (N,4)
            if len(boxes) == 0:
                continue

            # Grid centres in original image coordinates
            ys = (torch.arange(H, device=device).float() + 0.5) * stride
            xs = (torch.arange(W, device=device).float() + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            # (H,W)

            for box in boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1

                if bw <= 0 or bh <= 0:
                    continue

                # Assign positive locations within centre region
                # (FCOS centerness region = 0.5 * box size)
                r = 0.5
                in_box = (
                    (grid_x >= cx - r * bw) &
                    (grid_x <= cx + r * bw) &
                    (grid_y >= cy - r * bh) &
                    (grid_y <= cy + r * bh)
                )

                if not in_box.any():
                    # Fall back: closest grid point
                    dist = ((grid_x - cx)**2 +
                            (grid_y - cy)**2).sqrt()
                    min_idx = dist.argmin()
                    iy = min_idx // W
                    ix = min_idx  % W
                    in_box[iy, ix] = True

                cls_map[b, 0][in_box]    = 1.0
                pos_mask[b][in_box]      = True

                # ltrb regression targets at positive locations
                l = grid_x[in_box] - x1
                t = grid_y[in_box] - y1
                r_ = x2 - grid_x[in_box]
                b_ = y2 - grid_y[in_box]
                ltrb = torch.stack([l, t, r_, b_], dim=1)  # (K,4)

                # Centerness = sqrt((min(l,r)/max(l,r)) *
                #                   (min(t,b)/max(t,b)))
                ctr = torch.sqrt(
                    (torch.min(l, r_) / (torch.max(l, r_) + 1e-8)) *
                    (torch.min(t, b_) / (torch.max(t, b_) + 1e-8))
                ).clamp(0, 1)

                pos_indices = in_box.nonzero(as_tuple=False)
                for k, (py, px) in enumerate(pos_indices):
                    reg_map[b, :, py, px]    = ltrb[k]
                    center_map[b, 0, py, px] = ctr[k]

        targets.append({
            'cls_map'   : cls_map,
            'reg_map'   : reg_map,
            'center_map': center_map,
            'pos_mask'  : pos_mask,
        })

    return targets


# ── Synthetic dataloader for testing without real dataset ──────────────
class SyntheticDataset(torch.utils.data.Dataset):
    """
    Generates synthetic RGB+Thermal batches for pipeline testing.
    Used when KAIST dataset is not yet downloaded.
    Replace with KAISTDataset once data is ready.
    """

    def __init__(self, length=100, img_h=512, img_w=640):
        self.length = length
        self.h = img_h
        self.w = img_w

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random fault injection
        rgb_fault = 1 if torch.rand(1).item() < 0.15 else 0
        thr_fault = 1 if torch.rand(1).item() < 0.15 else 0

        rgb     = torch.randn(3, self.h, self.w)
        thermal = torch.randn(1, self.h, self.w)

        if rgb_fault:
            rgb = torch.zeros_like(rgb)
        if thr_fault:
            thermal = torch.zeros_like(thermal)

        # Random boxes (0-3 per image)
        n_boxes = torch.randint(0, 4, (1,)).item()
        if n_boxes > 0:
            x1 = torch.randint(0, self.w//2, (n_boxes,)).float()
            y1 = torch.randint(0, self.h//2, (n_boxes,)).float()
            x2 = (x1 + torch.randint(20, 100, (n_boxes,)).float()
                  ).clamp(max=self.w-1)
            y2 = (y1 + torch.randint(40, 150, (n_boxes,)).float()
                  ).clamp(max=self.h-1)
            boxes  = torch.stack([x1,y1,x2,y2], dim=1)
            labels = torch.zeros(n_boxes, dtype=torch.long)
        else:
            boxes  = torch.zeros((0,4))
            labels = torch.zeros((0,), dtype=torch.long)

        # Segmentation mask
        seg_mask = torch.randint(0, 2, (self.h, self.w))

        return {
            'rgb'           : rgb,
            'thermal'       : thermal,
            'boxes'         : boxes,
            'labels'        : labels,
            'seg_mask'      : seg_mask,
            'rgb_fault'     : torch.tensor(rgb_fault, dtype=torch.float32),
            'thermal_fault' : torch.tensor(thr_fault, dtype=torch.float32),
        }


def synthetic_collate(batch):
    return {
        'rgb'           : torch.stack([b['rgb']          for b in batch]),
        'thermal'       : torch.stack([b['thermal']      for b in batch]),
        'boxes'         : [b['boxes']                    for b in batch],
        'labels'        : [b['labels']                   for b in batch],
        'seg_mask'      : torch.stack([b['seg_mask']     for b in batch]),
        'rgb_fault'     : torch.stack([b['rgb_fault']    for b in batch]),
        'thermal_fault' : torch.stack([b['thermal_fault']for b in batch]),
    }


# ── Training epoch ─────────────────────────────────────────────────────
def train_one_epoch(
    model      : AFTMSFModel,
    loader     : DataLoader,
    optimizer  : torch.optim.Optimizer,
    loss_fn    : MultiTaskLoss,
    rel_reg    : ReliabilityRegulariser,
    scaler     : GradScaler,
    device     : torch.device,
    epoch      : int,
    writer     : SummaryWriter,
    log_every  : int = 10,
):
    model.train()
   # torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    total_loss_sum = 0.0
    det_loss_sum   = 0.0
    seg_loss_sum   = 0.0
    rel_loss_sum   = 0.0
    n_batches      = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        rgb     = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)
        seg_gt  = batch['seg_mask'].to(device)
# Downsample seg target to match stride-4 output
        seg_gt_ds = F.interpolate(
            seg_gt.unsqueeze(1).float(),
            size=(rgb.shape[2]//4, rgb.shape[3]//4),
            mode='nearest'
        ).squeeze(1).long()
        

        model_input = {'rgb': rgb, 'thermal': thermal}

        # Convert YOLO boxes (cx,cy,w,h norm) → (x1,y1,x2,y2) absolute
        H_img = rgb.shape[2]
        W_img = rgb.shape[3]
        boxes_abs = []
        for boxes in batch['boxes']:
            if len(boxes) == 0:
                boxes_abs.append(boxes)
                continue
            cx = boxes[:, 0] * W_img
            cy = boxes[:, 1] * H_img
            bw = boxes[:, 2] * W_img
            bh = boxes[:, 3] * H_img
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes_abs.append(torch.stack([x1,y1,x2,y2], dim=1))

        # Ground truth fault flags for reliability supervision
        fault_gt = {
            'rgb'    : batch['rgb_fault'].to(device),
            'thermal': batch['thermal_fault'].to(device),
        }

        # Forward with AMP
        with autocast('cuda'):
            output = model(
                model_input,
                target_size=None,
            )

            # Build detection targets
            pred_shapes = [
                (p['cls'].shape[2], p['cls'].shape[3])
                for p in output['det_preds']
            ]
            det_targets = build_det_targets(
                boxes_abs, batch['labels'],
                pred_shapes, device
            )

            # Reliability regularisation loss
            rel_loss = rel_reg(output['confidence'], fault_gt)

            # Total loss
            targets = {
                'det_targets'     : det_targets,
                'seg_target'      : seg_gt_ds,
                'reliability_loss': rel_loss,
            }
            loss_dict = loss_fn(output, targets)
            loss      = loss_dict['total']

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping — important for transformer stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss_sum += loss_dict['total'].item()
        det_loss_sum   += loss_dict['det'].item()
        seg_loss_sum   += loss_dict['seg'].item()
        rel_loss_sum   += loss_dict['rel'].item()
        n_batches      += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total'].item():.3f}",
            'det' : f"{loss_dict['det'].item():.3f}",
            'seg' : f"{loss_dict['seg'].item():.3f}",
        })

        # TensorBoard logging
        global_step = (epoch - 1) * len(loader) + batch_idx
        if batch_idx % log_every == 0:
            writer.add_scalar('train/total_loss',
                              loss_dict['total'].item(), global_step)
            writer.add_scalar('train/det_loss',
                              loss_dict['det'].item(),   global_step)
            writer.add_scalar('train/seg_loss',
                              loss_dict['seg'].item(),   global_step)
            writer.add_scalar('train/rel_loss',
                              loss_dict['rel'].item(),   global_step)

            # Log confidence scores
            for mod, conf in output['confidence'].items():
                writer.add_scalar(
                    f'train/confidence_{mod}',
                    conf.mean().item(), global_step
                )

            # Log fusion weights
            for mod, w in output['acaf_weights'].items():
                writer.add_scalar(
                    f'train/fusion_weight_{mod}',
                    w.mean().item(), global_step
                )

    avg = {
        'total': total_loss_sum / n_batches,
        'det'  : det_loss_sum   / n_batches,
        'seg'  : seg_loss_sum   / n_batches,
        'rel'  : rel_loss_sum   / n_batches,
    }
    return avg


# ── Main training function ─────────────────────────────────────────────
def train(config_path: str, stage: int = 1, resume: str = None,
          exp_dir: str = None, epochs: int = None):

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Stage  : {stage}")

    # Experiment directory
    exp_dir = Path(exp_dir) if exp_dir else Path('experiments') / f'stage{stage}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(exp_dir / 'tb_logs'))

    # ── Model ──────────────────────────────────────────────────────────
    model = AFTMSFModel(
        modalities     =['rgb', 'thermal'],
        num_det_classes=cfg['dataset']['num_classes'],
        num_seg_classes=cfg['dataset']['num_classes'] + 1,
        fpn_channels   =256,
        num_heads      =4,
        fault_thresh   =0.15,
        grad_checkpoint=True,
    ).to(device)

    # Free cache before training
    torch.cuda.empty_cache()

    model.set_train_stage(stage)

    total, trainable = model.count_parameters()
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # ── Optimiser ──────────────────────────────────────────────────────
    # Stage-specific learning rates
    base_lr = {1: 1e-3, 2: 5e-4, 3: 1e-4}.get(stage, 1e-4)

    param_groups = model.get_parameter_groups(base_lr)
    optimizer    = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-4,
    )

    # Cosine annealing scheduler
    epochs = epochs or {1: 10, 2: 10, 3: 30}.get(stage, 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=base_lr * 0.01
    )

    # AMP scaler
    scaler = GradScaler('cuda')

    # Resume from checkpoint
    start_epoch = 1
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        # Skip optimizer state when changing stages
        # (parameter groups change between stages)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"Resumed from epoch {ckpt['epoch']}")
        except ValueError:
            print(f"Optimizer state skipped (stage change) — "
                  f"model weights loaded from epoch {ckpt['epoch']}")
        start_epoch = 1

    # ── Loss functions ─────────────────────────────────────────────────
    loss_fn = MultiTaskLoss(
        lambda_det=1.0,
        lambda_seg=1.0,
        lambda_rel=0.1,
    ).to(device)

    rel_reg = ReliabilityRegulariser(
        modalities  =['rgb', 'thermal'],
        weight      =0.1,
        label_smooth=0.1,
    )

    # ── Dataset ────────────────────────────────────────────────────────
    # Use SyntheticDataset until KAIST download completes
    # Replace with KAISTDataset when data is ready:
    #
    # from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn
    # train_ds = KAISTDataset('configs/kaist.yaml', split='train')
    # loader   = DataLoader(train_ds, batch_size=4,
    #                       collate_fn=kaist_collate_fn,
    #                       num_workers=2, pin_memory=True)

    
    from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn
    train_ds = KAISTDataset(config_path, split='train', fault_inject=True)
    loader   = DataLoader(
        train_ds,
        batch_size =1 if stage >= 2 else cfg['dataloader']['batch_size'],
        shuffle    =True,
        num_workers=0,
        collate_fn =kaist_collate_fn,
        pin_memory =True,
    )

    print(f"Dataset: {len(train_ds)} samples, "
          f"batch_size={cfg['dataloader']['batch_size']}")
    print(f"Training for {epochs} epochs\n")

    # ── Training loop ──────────────────────────────────────────────────
    best_loss = float('inf')

    for epoch in range(start_epoch, epochs + 1):
        t0  = time.time()
        avg = train_one_epoch(
            model, loader, optimizer, loss_fn,
            rel_reg, scaler, device, epoch, writer,
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={avg['total']:.4f} | "
            f"det={avg['det']:.4f} | "
            f"seg={avg['seg']:.4f} | "
            f"rel={avg['rel']:.4f} | "
            f"lr={optimizer.param_groups[1]['lr']:.2e} | "
            f"time={elapsed:.1f}s"
        )

        # Log epoch metrics
        writer.add_scalar('epoch/total_loss', avg['total'], epoch)
        writer.add_scalar('epoch/lr',
                          optimizer.param_groups[1]['lr'], epoch)

        # Save checkpoint
        ckpt = {
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss'     : avg['total'],
            'config'   : cfg,
        }
        torch.save(ckpt, exp_dir / 'last.pth')

        if avg['total'] < best_loss:
            best_loss = avg['total']
            torch.save(ckpt, exp_dir / 'best.pth')
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")

    writer.close()
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {exp_dir}")


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/kaist.yaml')
    parser.add_argument('--stage',  type=int, default=1)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--exp_dir', default=None,
                        help='output folder (default experiments/stage<N>)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the stage epoch count (e.g. 1 for a demo)')
    args = parser.parse_args()

    train(args.config, args.stage, args.resume, args.exp_dir, args.epochs)
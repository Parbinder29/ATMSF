"""
Visualisation Script for AFTMSF System
Generates all figures needed for the paper:

    Fig 3 — Attention weight maps
    Fig 4 — Confidence score plots under fault conditions
    Fig 5 — Detection + segmentation overlay
    Fig 6 — Fusion weight distribution
    Fig 7 — Qualitative results (normal vs faulty)

Usage:
    python -m src.visualize --config configs/kaist.yaml
                            --checkpoint experiments/stage3/best.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.aftmsf_model import AFTMSFModel
from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn
from src.evaluate import decode_fcos_predictions


# ── Colour palette ─────────────────────────────────────────────────────
COLOURS = {
    'rgb'    : '#2196F3',   # blue
    'thermal': '#FF5722',   # orange
    'depth'  : '#4CAF50',   # green
    'normal' : '#4CAF50',
    'fault'  : '#F44336',
}


# ── Helper: denormalise image for display ──────────────────────────────
def denorm_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalised RGB tensor (3,H,W) → uint8 HWC numpy array.
    """
    mean = np.array([0.3465, 0.3219, 0.2842])
    std  = np.array([0.2358, 0.2265, 0.2274])
    img  = tensor.cpu().numpy().transpose(1,2,0)
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def denorm_thermal(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalised thermal tensor (1,H,W) → uint8 HW numpy array.
    """
    mean = 0.1598
    std  = 0.0813
    img  = tensor.cpu().numpy().squeeze()
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ── Fig 1: Detection + Segmentation overlay ───────────────────────────
def visualise_predictions(
    rgb_img   : np.ndarray,
    thr_img   : np.ndarray,
    det_preds : list,
    seg_logits: torch.Tensor,
    save_path : str,
    title     : str = "AFTMSF Predictions",
):
    """
    Save side-by-side: RGB with boxes | Thermal | Segmentation mask.
    """
    H, W  = rgb_img.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # RGB + detection boxes
    axes[0].imshow(rgb_img)
    axes[0].set_title('RGB + Detection')
    axes[0].axis('off')

    if len(det_preds) > 0 and len(det_preds[0]) > 0:
        boxes = det_preds[0].cpu().numpy()
        for box in boxes:
            x1,y1,x2,y2,score,cls = box
            rect = patches.Rectangle(
                (x1,y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='#FF0000',
                facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(
                x1, y1-5, f'{score:.2f}',
                color='white', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='red', alpha=0.7)
            )

    # Thermal
    axes[1].imshow(thr_img, cmap='inferno')
    axes[1].set_title('Thermal (LWIR)')
    axes[1].axis('off')

    # Segmentation
    seg_mask = seg_logits[0].argmax(0).cpu().numpy()
    seg_vis  = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
    seg_vis[seg_mask==0] = [50,  50,  50]   # background — dark
    seg_vis[seg_mask==1] = [255, 100, 0]    # person — orange

    # Resize to match input
    seg_vis = cv2.resize(seg_vis, (W, H),
                         interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(rgb_img, 0.6, seg_vis, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Segmentation Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Fig 2: Confidence scores under different fault conditions ──────────
def visualise_confidence_under_faults(
    model     : AFTMSFModel,
    sample_rgb: torch.Tensor,
    sample_thr: torch.Tensor,
    device    : torch.device,
    save_path : str,
):
    """
    Bar chart showing confidence scores under 5 conditions:
        Normal | RGB dropout | Thermal dropout |
        RGB noise | Thermal blur
    Paper Figure 4.
    """
    conditions = {
        'Normal'        : (sample_rgb, sample_thr),
        'RGB\nDropout'  : (torch.zeros_like(sample_rgb), sample_thr),
        'Thermal\nDropout': (sample_rgb, torch.zeros_like(sample_thr)),
        'RGB\nNoise'    : (
            sample_rgb + torch.randn_like(sample_rgb)*0.3,
            sample_thr
        ),
        'Thermal\nBlur' : (
            sample_rgb,
            F.avg_pool2d(
                sample_thr.unsqueeze(0),
                kernel_size=15, stride=1, padding=7
            ).squeeze(0)
        ),
    }

    rgb_confs = []
    thr_confs = []
    labels    = []

    model.eval()
    with torch.no_grad():
        for label, (rgb, thr) in conditions.items():
            batch = {
                'rgb'    : rgb.unsqueeze(0).to(device),
                'thermal': thr.unsqueeze(0).to(device),
            }
            out = model(batch, target_size=None)
            rgb_confs.append(
                out['confidence']['rgb'].mean().item()
            )
            thr_confs.append(
                out['confidence']['thermal'].mean().item()
            )
            labels.append(label)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, rgb_confs, width,
                   label='RGB', color=COLOURS['rgb'],
                   alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, thr_confs, width,
                   label='Thermal', color=COLOURS['thermal'],
                   alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0.15, color='red', linestyle='--',
               linewidth=1.5, label='Fault threshold (0.15)')

    ax.set_xlabel('Fault Condition', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title(
        'Per-Modality Confidence Scores Under Different Fault Conditions\n'
        '(AFTMSF Fault Detection Module)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Fig 3: Fusion weight distribution ─────────────────────────────────
def visualise_fusion_weights(
    model     : AFTMSFModel,
    sample_rgb: torch.Tensor,
    sample_thr: torch.Tensor,
    device    : torch.device,
    save_path : str,
):
    """
    Line plot showing how fusion weights change as
    thermal confidence is degraded from 1.0 to 0.0.
    Paper Figure 5.
    """
    conf_levels = np.linspace(0.0, 1.0, 20)
    rgb_weights = []
    thr_weights = []

    model.eval()
    with torch.no_grad():
        for conf_level in conf_levels:
            # Manually set confidence scores to test weight redistribution
            fake_conf = {
                'rgb'    : torch.tensor([0.9]).to(device),
                'thermal': torch.tensor([conf_level]).to(device),
            }
            weights = model._get_fusion_weights(fake_conf)
            rgb_weights.append(weights['rgb'].mean().item())
            thr_weights.append(weights['thermal'].mean().item())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(conf_levels, rgb_weights,
            color=COLOURS['rgb'], linewidth=2.5,
            label='RGB fusion weight', marker='o', markersize=4)
    ax.plot(conf_levels, thr_weights,
            color=COLOURS['thermal'], linewidth=2.5,
            label='Thermal fusion weight', marker='s', markersize=4)

    ax.axvline(x=0.15, color='red', linestyle='--',
               linewidth=1.5, label='Fault threshold (0.15)')
    ax.fill_between(conf_levels, 0, 1,
                    where=[c < 0.15 for c in conf_levels],
                    alpha=0.1, color='red', label='Fault region')

    ax.set_xlabel('Thermal Confidence Score', fontsize=12)
    ax.set_ylabel('Fusion Weight', fontsize=12)
    ax.set_title(
        'Adaptive Fusion Weight Redistribution\n'
        'as Thermal Sensor Confidence Degrades',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Fig 4: Training loss curves ───────────────────────────────────────
def plot_training_curves(save_path: str):
    """
    Plot training loss curves from logged data.
    Paper Figure 3.
    """
    # Stage 1 results
    stage1_epochs = list(range(1, 11))
    stage1_loss   = [9.88, 6.50, 5.81, 5.43, 4.99,
                     4.68, 4.41, 4.16, 3.98, 4.02]
    stage1_seg    = [0.199, 0.119, 0.104, 0.097, 0.086,
                     0.075, 0.071, 0.066, 0.060, 0.062]

    # Stage 2 results
    stage2_epochs = list(range(11, 15))
    stage2_loss   = [3.60, 2.81, 2.32, 2.03]
    stage2_seg    = [0.092, 0.070, 0.054, 0.043]

    # Stage 3 results (partial)
    stage3_epochs = list(range(15, 18))
    stage3_loss   = [1.59, 1.41, 1.30]
    stage3_seg    = [0.036, 0.030, 0.027]

    all_epochs = stage1_epochs + stage2_epochs + stage3_epochs
    all_loss   = stage1_loss   + stage2_loss   + stage3_loss
    all_seg    = stage1_seg    + stage2_seg    + stage3_seg

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('AFTMSF Training Convergence',
                 fontsize=14, fontweight='bold')

    # Total loss
    ax1.plot(stage1_epochs, stage1_loss, 'b-o',
             markersize=4, label='Stage 1 (frozen encoders)')
    ax1.plot(stage2_epochs, stage2_loss, 'g-o',
             markersize=4, label='Stage 2 (layer4 unfrozen)')
    ax1.plot(stage3_epochs, stage3_loss, 'r-o',
             markersize=4, label='Stage 3 (full fine-tune)')
    ax1.axvline(x=10.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=14.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Seg loss
    ax2.plot(stage1_epochs, stage1_seg, 'b-o',
             markersize=4, label='Stage 1')
    ax2.plot(stage2_epochs, stage2_seg, 'g-o',
             markersize=4, label='Stage 2')
    ax2.plot(stage3_epochs, stage3_seg, 'r-o',
             markersize=4, label='Stage 3')
    ax2.axvline(x=10.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=14.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Segmentation Loss', fontsize=12)
    ax2.set_title('Segmentation Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────
def visualize(config_path: str, checkpoint: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path('experiments/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = AFTMSFModel(
        modalities     =['rgb', 'thermal'],
        num_det_classes=cfg['dataset']['num_classes'],
        num_seg_classes=cfg['dataset']['num_classes'] + 1,
        fpn_channels   =256,
        num_heads      =4,
        fault_thresh   =0.15,
        grad_checkpoint=False,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model loaded — epoch {ckpt['epoch']}, "
          f"loss={ckpt['loss']:.4f}")

    # Load a few val samples
    dataset = KAISTDataset(config_path, split='val',
                           fault_inject=False)
    loader  = DataLoader(dataset, batch_size=1,
                         shuffle=False, num_workers=0,
                         collate_fn=kaist_collate_fn)

    # Get one sample
    batch   = next(iter(loader))
    rgb     = batch['rgb'].to(device)
    thermal = batch['thermal'].to(device)

    with torch.no_grad():
        output = model(
            {'rgb': rgb, 'thermal': thermal},
            target_size=None
        )

    det_preds = decode_fcos_predictions(
        output['det_preds'],
        rgb.shape[2], rgb.shape[3],
        score_thr=0.3,
    )

    rgb_img = denorm_rgb(rgb[0])
    thr_img = denorm_thermal(thermal[0])

    # Generate all figures
    print("\nGenerating figures...")

    # Fig 1 — Predictions
    visualise_predictions(
        rgb_img, thr_img, det_preds,
        output['seg_logits'],
        str(out_dir / 'fig1_predictions.png'),
        title='AFTMSF: Detection + Segmentation Results',
    )

    # Fig 2 — Confidence under faults
    visualise_confidence_under_faults(
        model, rgb[0], thermal[0], device,
        str(out_dir / 'fig2_confidence_faults.png'),
    )

    # Fig 3 — Fusion weight redistribution
    visualise_fusion_weights(
        model, rgb[0], thermal[0], device,
        str(out_dir / 'fig3_fusion_weights.png'),
    )

    # Fig 4 — Training curves
    plot_training_curves(
        str(out_dir / 'fig4_training_curves.png'),
    )

    print(f"\nAll figures saved to {out_dir}/")
    print("Use these directly in your IEEE paper and B.Tech report.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/kaist.yaml')
    parser.add_argument('--checkpoint',
                        default='experiments/stage3/best.pth')
    args = parser.parse_args()
    visualize(args.config, args.checkpoint)
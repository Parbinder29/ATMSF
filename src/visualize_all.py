"""
Comprehensive Visualisation Script for AFTMSF Paper
Generates all figures needed for IEEE TITS paper and B.Tech report

Figures generated:
    Fig 1  — Training loss curves (all 3 datasets)
    Fig 2  — Dataset comparison bar chart
    Fig 3  — Baseline comparison
    Fig 4  — Fault injection heatmap
    Fig 5  — Confidence scores under faults
    Fig 6  — Fusion weight redistribution
    Fig 7  — Per-stage loss breakdown
    Fig 8  — FPS vs mAP tradeoff
    Fig 9  — mIoU breakdown per class
    Fig 10 — Learning rate schedule

Usage:
    python -m src.visualize_all
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path('experiments/figures_paper')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fault results produced by src.experiments.fault_experiment (seed 0)
FAULT_RESULTS = Path('experiments/fault_experiment_results.json')


def load_fault_results():
    with open(FAULT_RESULTS) as f:
        return json.load(f)

# ── Colour scheme ──────────────────────────────────────────────────────
C = {
    'kaist' : '#2196F3',   # blue
    'kitti' : '#4CAF50',   # green
    'flir'  : '#FF9800',   # orange
    'aftmsf': '#E91E63',   # pink/red
    'early' : '#9C27B0',   # purple
    'late'  : '#00BCD4',   # cyan
    'rgb'   : '#F44336',   # red
    'stage1': '#1565C0',
    'stage2': '#2E7D32',
    'stage3': '#B71C1C',
}

plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi'   : 150,
    'axes.grid'    : True,
    'grid.alpha'   : 0.3,
})


# ======================================================================
# Fig 1 — Training loss curves (all 3 datasets + all stages)
# ======================================================================
def fig1_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('AFTMSF Training Convergence Across All Datasets',
                 fontsize=14, fontweight='bold')

    # KAIST
    s1 = [9.88,6.50,5.81,5.43,4.99,4.68,4.41,4.16,3.98,4.02]
    s2 = [3.60,2.81,2.32,2.03]
    s3 = [1.59,1.41,1.30,1.23,1.20,1.11,1.08,0.89,0.78,0.72,
          0.71,0.69,0.65,0.63,0.60,0.57,0.59,0.56,0.58,0.56,
          0.65,0.63,0.61,0.58,0.60,0.57,0.58,0.56,0.58,0.56]
    e1=list(range(1,11)); e2=list(range(11,15)); e3=list(range(15,45))
    ax = axes[0]
    ax.plot(e1,s1,'o-',color=C['stage1'],label='Stage 1',markersize=3)
    ax.plot(e2,s2,'s-',color=C['stage2'],label='Stage 2',markersize=3)
    ax.plot(e3,s3[:30],'^-',color=C['stage3'],
            label='Stage 3',markersize=2)
    ax.axvline(10.5,color='gray',linestyle='--',alpha=0.5)
    ax.axvline(14.5,color='gray',linestyle='--',alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('KAIST (RGB+Thermal)')
    ax.legend(); ax.set_ylim(0, 12)

    # KITTI
    ks1 = [11.96,8.92,8.13,7.55,7.12]
    ks2 = [6.95,6.54,6.15,5.97,5.81]
    ke1 = list(range(1,6)); ke2 = list(range(6,11))
    ax = axes[1]
    ax.plot(ke1,ks1,'o-',color=C['stage1'],label='Stage 1',markersize=4)
    ax.plot(ke2,ks2,'s-',color=C['stage2'],label='Stage 2',markersize=4)
    ax.axvline(5.5,color='gray',linestyle='--',alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('KITTI (RGB+LiDAR Depth)')
    ax.legend(); ax.set_ylim(0, 15)

    # FLIR
    fs1 = [4.78,4.30,4.06,3.97,3.88]
    fs2 = [3.20,3.03,2.92,2.79,2.72]
    fs3 = [2.72,2.64,2.58,2.53,2.46,2.42,2.36,2.33,2.30,2.27]
    fe1=list(range(1,6)); fe2=list(range(6,11)); fe3=list(range(11,21))
    ax = axes[2]
    ax.plot(fe1,fs1,'o-',color=C['stage1'],label='Stage 1',markersize=4)
    ax.plot(fe2,fs2,'s-',color=C['stage2'],label='Stage 2',markersize=4)
    ax.plot(fe3,fs3,'^-',color=C['stage3'],label='Stage 3',markersize=4)
    ax.axvline(5.5, color='gray',linestyle='--',alpha=0.5)
    ax.axvline(10.5,color='gray',linestyle='--',alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('FLIR (RGB+Thermal)')
    ax.legend(); ax.set_ylim(0, 6)

    plt.tight_layout()
    path = OUT_DIR / 'fig1_training_curves_all_datasets.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 2 — Baseline comparison bar chart
# ======================================================================
def fig2_baseline_comparison():
    methods = ['RGB\nOnly', 'Early\nFusion', 'Late\nFusion',
               'AFTMSF\n(Ours)']
    map_vals  = [69.97, 70.66, 77.89, 89.14]
    miou_vals = [37.00, 38.45, 36.34, 44.28]

    x     = np.arange(len(methods))
    width = 0.35
    colors_map  = [C['rgb'], C['early'], C['late'], C['aftmsf']]
    colors_miou = [c+'99' for c in colors_map]  # lighter

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, map_vals,  width,
                   label='mAP@0.5 (%)',
                   color=colors_map, edgecolor='black',
                   linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, miou_vals, width,
                   label='mIoU (%)',
                   color=colors_map, edgecolor='black',
                   linewidth=0.5, alpha=0.5, hatch='//')

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5,
                f'{h:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5,
                f'{h:.1f}', ha='center', va='bottom',
                fontsize=9)

    # Highlight AFTMSF
    ax.axhline(y=89.14, color=C['aftmsf'], linestyle='--',
               linewidth=1.5, alpha=0.7, label='AFTMSF mAP baseline')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('AFTMSF vs Baseline Methods — KAIST Test Set',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    path = OUT_DIR / 'fig2_baseline_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 3 — Fault injection heatmap
# ======================================================================
def fig3_fault_heatmap():
    fault_modes = ['Dropout', 'Blur', 'Occlusion', 'Gaussian\nNoise']
    modalities  = ['RGB', 'Thermal']

    # mAP under each fault, read from fault_experiment_results.json
    res   = load_fault_results()
    modes = ['dropout', 'blur', 'occlusion', 'gaussian_noise']
    map_vals = np.array([
        [res[f'{m}_{mode}']['mAP@0.5'] for mode in modes]
        for m in ('rgb', 'thermal')
    ])
    drops = res['baseline']['mAP@0.5'] - map_vals

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fault Injection Experiment Results',
                 fontsize=14, fontweight='bold')

    # mAP heatmap
    im1 = axes[0].imshow(map_vals, cmap='RdYlGn',
                          vmin=60, vmax=92, aspect='auto')
    axes[0].set_xticks(range(len(fault_modes)))
    axes[0].set_xticklabels(fault_modes)
    axes[0].set_yticks(range(len(modalities)))
    axes[0].set_yticklabels(modalities, fontsize=12)
    axes[0].set_title('mAP@0.5 (%) Under Fault Conditions')
    plt.colorbar(im1, ax=axes[0])
    for i in range(len(modalities)):
        for j in range(len(fault_modes)):
            axes[0].text(j, i, f'{map_vals[i,j]:.1f}%',
                        ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='black')

    # Drop heatmap
    im2 = axes[1].imshow(drops, cmap='Reds',
                          vmin=0, vmax=30, aspect='auto')
    axes[1].set_xticks(range(len(fault_modes)))
    axes[1].set_xticklabels(fault_modes)
    axes[1].set_yticks(range(len(modalities)))
    axes[1].set_yticklabels(modalities, fontsize=12)
    axes[1].set_title('mAP Drop (pp) from Baseline 89.14%')
    plt.colorbar(im2, ax=axes[1])
    for i in range(len(modalities)):
        for j in range(len(fault_modes)):
            axes[1].text(j, i, f'-{drops[i,j]:.1f}',
                        ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if drops[i,j]>15 else 'black')

    plt.tight_layout()
    path = OUT_DIR / 'fig3_fault_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 4 — Cross-dataset performance comparison
# ======================================================================
def fig4_cross_dataset():
    datasets = ['KAIST\n(RGB+Thermal)', 'KITTI\n(RGB+Depth)',
                'FLIR\n(RGB+Thermal)']
    map_vals  = [89.14, 21.53, 1.52]
    miou_vals = [44.28, 22.55, 16.34]
    fps_vals  = [13.9,  31.6,  36.8 ]

    x     = np.arange(len(datasets))
    width = 0.25
    colors = [C['kaist'], C['kitti'], C['flir']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Cross-Dataset Generalisation of AFTMSF',
                 fontsize=14, fontweight='bold')

    # mAP
    bars = axes[0].bar(x, map_vals, width*2,
                       color=colors, edgecolor='black',
                       linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, map_vals):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+1,
                     f'{v:.1f}%', ha='center', va='bottom',
                     fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel('mAP@0.5 (%)')
    axes[0].set_title('Detection Performance')
    axes[0].set_ylim(0, 100)

    # mIoU
    bars = axes[1].bar(x, miou_vals, width*2,
                       color=colors, edgecolor='black',
                       linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, miou_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.5,
                     f'{v:.1f}%', ha='center', va='bottom',
                     fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel('mIoU (%)')
    axes[1].set_title('Segmentation Performance')
    axes[1].set_ylim(0, 60)

    # FPS
    bars = axes[2].bar(x, fps_vals, width*2,
                       color=colors, edgecolor='black',
                       linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, fps_vals):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.3,
                     f'{v:.1f}', ha='center', va='bottom',
                     fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(datasets)
    axes[2].set_ylabel('FPS')
    axes[2].set_title('Inference Speed')

    plt.tight_layout()
    path = OUT_DIR / 'fig4_cross_dataset.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 5 — Per-stage loss breakdown (det/seg/rel)
# ======================================================================
def fig5_loss_breakdown():
    # KAIST Stage 3 epoch-by-epoch breakdown
    epochs = list(range(1, 31))
    det_loss = [1.55,1.38,1.27,1.20,1.17,1.09,1.07,0.87,
                0.77,0.71,0.70,0.68,0.63,0.61,0.59,0.57,
                0.57,0.55,0.56,0.55,0.63,0.62,0.60,0.57,
                0.58,0.56,0.56,0.55,0.56,0.55]
    seg_loss = [0.036,0.030,0.027,0.027,0.027,0.024,0.024,
                0.020,0.016,0.014,0.014,0.014,0.016,0.015,
                0.014,0.013,0.013,0.014,0.012,0.011,0.014,
                0.014,0.012,0.012,0.013,0.012,0.013,0.012,
                0.013,0.011]
    rel_loss = [0.020,0.020,0.019,0.019,0.019,0.018,0.017,
                0.016,0.015,0.015,0.015,0.014,0.015,0.015,
                0.015,0.015,0.015,0.014,0.014,0.014,0.014,
                0.014,0.014,0.014,0.014,0.014,0.014,0.014,
                0.014,0.014]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(epochs, det_loss, '-', color='#1565C0',
            linewidth=2, label='Detection loss')
    ax.plot(epochs, seg_loss, '-', color='#2E7D32',
            linewidth=2, label='Segmentation loss (×10)')
    ax.plot(epochs, [r*10 for r in rel_loss], '-',
            color='#B71C1C', linewidth=2,
            label='Reliability loss (×10)')

    ax.fill_between(epochs, det_loss, alpha=0.1, color='#1565C0')
    ax.set_xlabel('Epoch (Stage 3)')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Component Breakdown During Stage 3 Training '
                 '(KAIST)', fontweight='bold')
    ax.legend()
    ax.set_xlim(1, 30)

    plt.tight_layout()
    path = OUT_DIR / 'fig5_loss_breakdown.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 6 — FPS vs mAP tradeoff
# ======================================================================
def fig6_fps_map_tradeoff():
    methods = ['RGB Only', 'Early Fusion', 'Late Fusion', 'AFTMSF']
    map_v   = [69.97, 70.66, 77.89, 89.14]
    fps_v   = [45.0,  40.0,  25.0,  13.9]
    params  = [27.3,  27.3,  54.1,  67.4]  # millions
    colors  = [C['rgb'], C['early'], C['late'], C['aftmsf']]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method, m, f, p, c) in enumerate(
        zip(methods, map_v, fps_v, params, colors)
    ):
        size = p * 5   # bubble size proportional to params
        ax.scatter(f, m, s=size, c=c, alpha=0.8,
                   edgecolors='black', linewidth=1.5,
                   zorder=5)
        offset = (2, 1) if method != 'Late Fusion' else (2, -3)
        ax.annotate(
            f'{method}\n({p}M params)',
            (f, m),
            xytext=(f+offset[0], m+offset[1]),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray',
                           lw=0.8),
        )

    ax.set_xlabel('Inference Speed (FPS)', fontsize=12)
    ax.set_ylabel('mAP@0.5 (%)', fontsize=12)
    ax.set_title('Performance vs Speed Tradeoff\n'
                 '(bubble size = model parameters)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 60)
    ax.set_ylim(60, 95)

    # Ideal region annotation
    ax.annotate('Ideal\nregion',
                xy=(50, 90), fontsize=10,
                color='green', alpha=0.7,
                ha='center')

    plt.tight_layout()
    path = OUT_DIR / 'fig6_fps_map_tradeoff.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 7 — Confidence score distribution
# ======================================================================
def fig7_confidence_distribution():
    conditions = ['Normal', 'RGB\nDropout', 'Thermal\nDropout',
                  'RGB\nNoise', 'Thermal\nBlur']
    res  = load_fault_results()
    keys = ['baseline', 'rgb_dropout', 'thermal_dropout',
            'rgb_gaussian_noise', 'thermal_blur']
    rgb_conf = [res[k]['rgb_conf']     for k in keys]
    thr_conf = [res[k]['thermal_conf'] for k in keys]

    x     = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, rgb_conf, width,
                   label='RGB confidence',
                   color=C['kaist'], alpha=0.85,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, thr_conf, width,
                   label='Thermal confidence',
                   color=C['flir'], alpha=0.85,
                   edgecolor='black', linewidth=0.5)

    ax.axhline(y=0.15, color='red', linestyle='--',
               linewidth=2, label='Fault threshold (τ=0.15)')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('Confidence Score c_i', fontsize=12)
    ax.set_title('Per-Modality Confidence Scores Under Fault Conditions\n'
                 'AFTMSF Fault Detection Module (KAIST Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    path = OUT_DIR / 'fig7_confidence_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 8 — Fusion weight redistribution
# ======================================================================
def fig8_fusion_weights():
    conf_levels = np.linspace(0, 1, 50)
    # Simulated fusion weights based on softmax(c/τ) with τ=1
    rgb_weights = []
    thr_weights = []
    for c in conf_levels:
        rgb_c = 0.9  # RGB confidence fixed
        thr_c = c
        w_rgb = np.exp(rgb_c) / (np.exp(rgb_c) + np.exp(thr_c))
        w_thr = np.exp(thr_c) / (np.exp(rgb_c) + np.exp(thr_c))
        rgb_weights.append(w_rgb)
        thr_weights.append(w_thr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(conf_levels, rgb_weights,
            color=C['kaist'], linewidth=2.5,
            label='RGB fusion weight', marker='o',
            markevery=5, markersize=5)
    ax.plot(conf_levels, thr_weights,
            color=C['flir'], linewidth=2.5,
            label='Thermal fusion weight', marker='s',
            markevery=5, markersize=5)

    ax.axvline(x=0.15, color='red', linestyle='--',
               linewidth=2, label='Fault threshold (τ=0.15)')
    ax.fill_between(conf_levels, 0, 1,
                    where=[c < 0.15 for c in conf_levels],
                    alpha=0.1, color='red', label='Fault region')

    ax.set_xlabel('Thermal Sensor Confidence Score', fontsize=12)
    ax.set_ylabel('Fusion Weight', fontsize=12)
    ax.set_title('Adaptive Fusion Weight Redistribution\n'
                 'as Thermal Confidence Degrades (ACAF Module)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    path = OUT_DIR / 'fig8_fusion_weights.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Fig 9 — Learning rate schedule
# ======================================================================
def fig9_lr_schedule():
    # Stage 1: 10 epochs, lr=1e-3, cosine
    # Stage 2: 10 epochs, lr=5e-4, cosine
    # Stage 3: 30 epochs, lr=1e-4, cosine
    def cosine_lr(base_lr, min_lr, t_max, epochs):
        return [min_lr + 0.5*(base_lr-min_lr) *
                (1 + np.cos(np.pi*e/t_max))
                for e in range(epochs)]

    lr_s1 = cosine_lr(1e-3, 1e-5, 10, 10)
    lr_s2 = cosine_lr(5e-4, 5e-6, 10, 10)
    lr_s3 = cosine_lr(1e-4, 1e-6, 30, 30)

    epochs_s1 = list(range(1, 11))
    epochs_s2 = list(range(11, 21))
    epochs_s3 = list(range(21, 51))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(epochs_s1, lr_s1, '-', color=C['stage1'],
            linewidth=2.5, label='Stage 1 (encoders frozen)')
    ax.plot(epochs_s2, lr_s2, '-', color=C['stage2'],
            linewidth=2.5, label='Stage 2 (layer4 unfrozen)')
    ax.plot(epochs_s3, lr_s3, '-', color=C['stage3'],
            linewidth=2.5, label='Stage 3 (full fine-tune)')

    ax.axvline(10.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(20.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(5,  1.1e-3, 'Stage 1', ha='center', fontsize=10,
            color=C['stage1'])
    ax.text(15, 5.5e-4, 'Stage 2', ha='center', fontsize=10,
            color=C['stage2'])
    ax.text(35, 1.1e-4, 'Stage 3', ha='center', fontsize=10,
            color=C['stage3'])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Three-Stage Training Curriculum — '
                 'Learning Rate Schedule (KAIST)',
                 fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = OUT_DIR / 'fig9_lr_schedule.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================================================================
# Main
# ======================================================================
if __name__ == '__main__':
    print("Generating all paper figures...")
    print(f"Output: {OUT_DIR}/\n")

    fig1_training_curves()
    fig2_baseline_comparison()
    fig3_fault_heatmap()
    fig4_cross_dataset()
    fig5_loss_breakdown()
    fig6_fps_map_tradeoff()
    fig7_confidence_distribution()
    fig8_fusion_weights()
    fig9_lr_schedule()

    print(f"\nAll {9} figures saved to {OUT_DIR}/")
    print("Ready for IEEE TITS paper and B.Tech report.")
    
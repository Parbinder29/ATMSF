# AFTMSF — Adaptive Fault-Tolerant Multi-Sensor Fusion

RGB + Thermal + LiDAR-depth fusion for autonomous perception. A three-signal fault detector
(autoencoder reconstruction error, feature variance, prediction entropy) estimates a confidence
score per sensor, and the ACAF module uses those scores to gate cross-modal attention, so a
failing sensor is down-weighted instead of corrupting the fused features. One model produces
FCOS-style detections and DeepLabV3+-style segmentation.

B.Tech Project-I, Group 23G15 — Sundram Kumar Ray, Arshiya, Parbinder Singh, Anupriya.
Supervisor: Prof. Dinesh Kumar, Dept. of CSE, Central University of Jammu.

## Why

Autonomous vehicles combine cameras, thermal imagers and LiDAR because each fails in different
conditions: RGB in darkness and glare, thermal on fine texture, LiDAR in rain and fog. Most fusion
networks weight every sensor the same way regardless of its state, so a silently failing sensor
feeds corrupted features into the fused representation. AFTMSF estimates each sensor's
reliability at runtime, with no fault labels needed at inference, and shifts fusion weight toward
the sensors that are still working.

## Architecture

```
 RGB image ──► ResNet-50 + FPN ──┐                      ┌──► FCOS detection head ──► boxes
                                 │                      │    (4 FPN scales)
 Thermal ───► ResNet-50 + FPN ───┼─► Fault detector ──► ACAF fusion
 (1-channel)                     │   c_i ∈ [0,1]        │
                                 │   per sensor         └──► DeepLabV3+ seg head ──► mask
 LiDAR depth ► DW-separable CNN ─┘
               + FPN (3.3M params)
```

| Component | What it does |
|---|---|
| **Encoders** | 256-channel FPN features at strides 4/8/16/32. The thermal ResNet-50 takes 1 channel, initialised by averaging the ImageNet RGB filters. The depth encoder fills sparse LiDAR returns with dilated convolutions first. |
| **Fault detector** | Per sensor, combines three signals with learned weights: autoencoder reconstruction error (input looks unlike training data), feature variance (dropout gives ~0, noise gives very high), and prediction entropy (uncertain predictions). Output: confidence `c_i`. |
| **ACAF fusion** | Fusion weights `w = softmax(c / τ)` with a learned temperature τ. Each sensor attends to all sensors with 4-head cross-attention on 8×8 pooled tokens (≈900× less memory than full attention, so it fits in 6 GB), and the outputs are summed with weights `w`. |
| **Task heads** | Anchor-free FCOS detection on all 4 scales; DeepLabV3+ segmentation with ASPP on the aggregated stride-4 feature. |

Training loss: `L = (focal + L1 + centerness) + (0.5·CE + 0.5·Dice) + 0.1·MSE(c_i, fault label)`.
The last term teaches the fault detector using synthetic faults (dropout, Gaussian noise,
occlusion, blur) injected during training.

Training runs in three stages on a single 6 GB GPU: (1) encoders frozen, lr 1e-3; (2) last encoder
block unfrozen, lr 5e-4; (3) full fine-tuning, lr 1e-4. AdamW, cosine schedule, FP16 mixed
precision, gradient clipping at 10.

Model size (KAIST configuration): 67.4M parameters — encoders 53.3M, ACAF 11.6M, detection head
4.7M, segmentation head 1.3M, fault detector 0.3M.

## Results (KAIST test set, 131 images)

| Method | Params | mAP@0.5 | mIoU |
|---|---|---|---|
| RGB Only | 27.3M | 69.97% | 37.00% |
| Early Fusion | 27.3M | 70.66% | 38.45% |
| Late Fusion | 54.1M | 77.89% | 36.34% |
| **AFTMSF** | 67.4M | **89.14%** | **44.28%** |

Fault injection (100% of test images corrupted, `--seed 0`):

| Fault | RGB mAP@0.5 | Thermal mAP@0.5 |
|---|---|---|
| None | 89.14% | 89.14% |
| Dropout | 63.85% | 78.20% |
| Blur | 69.83% | 79.41% |
| Occlusion | 78.77% | 81.03% |
| Gaussian noise (σ=0.3) | 80.82% | 80.61% |

Other datasets:

| Dataset | Modalities | Checkpoint | mAP@0.5 | mIoU |
|---|---|---|---|---|
| KAIST val | RGB + Thermal | `stage3/last.pth` | 89.71% | 47.87% |
| KITTI val (748) | RGB + LiDAR depth | `kitti_stage2/last.pth` | 21.53% | 22.55% |
| FLIR val (1,257) | RGB + Thermal | `flir_stage3/last.pth` | 1.52% | 16.34% |

FLIR detection is near zero because its objects average 6×15 px at 320×256, below what the
stride-4 FPN level can resolve. Segmentation still works.

Speed on an RTX 4050 Laptop GPU (6 GB) was 11–37 FPS across datasets; it varies with the
laptop's power mode, so plug in the charger and use the "Best performance" mode when timing.

## Setup

Windows, NVIDIA GPU with ≥ 6 GB VRAM. In the **Anaconda Prompt**:

```
conda create -n aftmsf python=3.10 -y
conda activate aftmsf
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install opencv-python-headless==4.11.0.86
```

The first run downloads the ImageNet ResNet-50 weights (~100 MB).

## Datasets (not in this repo)

Place them so the paths below exist. Dataset paths are hard-coded to `D:\aftmsf\data\...`
in `src/datasets/*.py`, `src/train_flir.py` and `src/evaluate_flir.py`; edit those if you use another location.

| Dataset | Source | Expected path |
|---|---|---|
| KAIST (YOLO early-fusion preview, 1,374 RGBA images) | Kaggle: `kaist-dataset-yolo26-early-fusion-preview-set` | `data/kaist/kaist_yolo_early/kaist_yolo_early/{images,labels}/{train,val,test}` |
| KITTI object detection | https://www.cvlibs.net/datasets/kitti/eval_object.php — left colour images, Velodyne point clouds, training labels | `data/kitti/training/{image_2,velodyne,label_2}` |
| FLIR ADAS v1.3 | https://www.flir.com/oem/adas/adas-dataset-form/ | `data/flir/FLIR_ADAS_1_3/{train,val}` |

## Trained checkpoints (not in this repo)

Checkpoints go in `experiments/`. Each one is 100–800 MB, over GitHub's file limit; they are
published separately (see the repository's Releases page).

| File | Used for |
|---|---|
| `experiments/stage3/last.pth` | KAIST — all main results and fault experiments |
| `experiments/kitti_stage2/last.pth` | KITTI results |
| `experiments/flir_stage3/last.pth` | FLIR results |
| `experiments/baselines/{RGBOnly,EarlyFusion,LateFusion}.pth` | Baseline comparison |

## Reproduce

```
python -m src.evaluate --config configs/kaist.yaml --checkpoint experiments/stage3/last.pth --split test
python -m src.evaluate_kitti --checkpoint experiments/kitti_stage2/last.pth
python -m src.evaluate_flir --checkpoint experiments/flir_stage3/last.pth
python -m src.experiments.fault_experiment --checkpoint experiments/stage3/last.pth --seed 0
python -m src.visualize_all
```

`visualize_all` writes the paper figures to `experiments/figures_paper/`; the fault heatmap
and confidence figures read `experiments/fault_experiment_results.json`.

## Train from scratch

```
python -m src.train --config configs/kaist.yaml --stage 1
python -m src.train --config configs/kaist.yaml --stage 2 --resume experiments/stage1/last.pth
python -m src.train --config configs/kaist.yaml --stage 3 --resume experiments/stage2/last.pth
python -m src.train_kitti --config configs/kitti.yaml --stage 1
python -m src.train_kitti --config configs/kitti.yaml --stage 2 --resume experiments/kitti_stage1/last.pth
python -m src.train_flir --config configs/flir.yaml --stage 1
python -m src.train_flir --config configs/flir.yaml --stage 2 --resume experiments/flir_stage1/last.pth
python -m src.train_flir --config configs/flir.yaml --stage 3 --resume experiments/flir_stage2/last.pth
python -m src.experiments.baseline_experiment
```

## Layout

```
configs/      kaist.yaml, kitti.yaml, flir.yaml
src/models/   encoders/ (rgb, thermal, depth), fault_detector, acaf_module,
              detection_head, seg_head, aftmsf_model, baselines
src/datasets/ kaist_dataset, kitti_dataset, flir_dataset (fault injection is built in)
src/losses/   multitask_loss, reliability_reg
src/experiments/ fault_experiment, baseline_experiment
src/          train*, evaluate*, visualize, visualize_all
experiments/  result files (*.json, *_eval.yaml), figures/, figures_paper/, TensorBoard logs
paper/        report and paper PDFs
```

## Limitations

- KAIST results use a 1,374-image subset of the KAIST benchmark, so they are not directly comparable with numbers reported on the full benchmark.
- FLIR detection is near zero because of very small objects; a stride-2 detection level would be needed.
- Images are trained at 256×320 (KAIST native is 512×640) because of the 6 GB GPU memory limit.

## Acknowledgements

Datasets: KAIST Multispectral Pedestrian (Hwang et al., CVPR 2015), KITTI (Geiger et al., IJRR 2013)
and FLIR ADAS v1.3 (Teledyne FLIR). Each dataset is subject to its own licence terms.
Built with PyTorch and torchvision.

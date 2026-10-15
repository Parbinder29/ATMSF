# AFTMSF — Commands for the Live Demo

Copy each command into the terminal one at a time and wait for it to finish.
All commands use the trained models already in `experiments\`, so nothing needs retraining.
Times are approximate on the RTX 4050 laptop (plugged in).

---

## Step 0 — Open the terminal (do this before the presentation starts)

1. Press the **Windows key**, type **Anaconda Prompt**, open **Anaconda Prompt (miniconda3)**.
2. Paste:

```
cd /d D:\aftmsf
conda activate aftmsf
set NO_ALBUMENTATIONS_UPDATE=1
```

The line should now start with `(aftmsf)`.

## Step 1 — Show the GPU and environment are working  (5 s)

```
python -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"
```

Expected: `PyTorch 2.6.0+cu124 | GPU: NVIDIA GeForce RTX 4050 Laptop GPU`

---

## Part A — Main results

### Step 2 — KAIST test set: detection + segmentation  (~1 min)

```
python -m src.evaluate --config configs/kaist.yaml --checkpoint experiments/stage3/last.pth --split test
```

Expected: **mAP@0.5 = 89.14%**, **mIoU = 44.28%**

### Step 3 — KAIST validation set  (~1 min)

```
python -m src.evaluate --config configs/kaist.yaml --checkpoint experiments/stage3/last.pth --split val
```

Expected: mAP@0.5 = 89.71%, mIoU = 47.87%

### Step 4 — KITTI (RGB + LiDAR depth)  (~2–3 min)

```
python -m src.evaluate_kitti --checkpoint experiments/kitti_stage2/last.pth
```

Expected: mAP@0.5 = 21.53%, mIoU = 22.55%

### Step 5 — FLIR (RGB + thermal)  (~3 min)

```
python -m src.evaluate_flir --checkpoint experiments/flir_stage3/last.pth
```

Expected: mAP@0.5 = 1.52%, mIoU = 16.34% (small-object limitation)

> FPS printed by Steps 2–5 includes data loading, so it is lower than the benchmark in Step 9.

---

## Part B — Fault tolerance

### Step 6 — Sensor fault injection on AFTMSF  (~3–4 min)

```
python -m src.experiments.fault_experiment --checkpoint experiments/stage3/last.pth --seed 0
```

Expected: clean 89.14% → RGB dropout **63.85%**, RGB blur 69.83%, RGB occlusion 78.77%,
RGB noise 80.82%, thermal dropout 78.20%, thermal blur 79.41%, thermal occlusion 81.03%,
thermal noise 80.61%.

### Step 7 — Same faults on the three baselines  (~3 min)

```
python -m src.experiments.baseline_fault_experiment
```

Expected: under RGB dropout, RGB Only, Early Fusion and Late Fusion all give **0.00%**.

### Step 8 — Fault detector ablation  (~5 min)

```
python -m src.experiments.ablation_experiment
```

Expected: full 89.14%, without entropy 89.31%, without fault detector 89.20%;
every fault changes by less than 1 pp with/without the detector.

### Step 9 — Speed and model size  (~3 min)

```
python -m src.benchmark
```

Expected: AFTMSF ≈ 22 FPS (KAIST), ≈ 26 (KITTI), ≈ 23 (FLIR); 67,350,042 parameters.

---

## Part C — Pictures

### Step 10 — Prediction examples (boxes, segmentation, confidence)  (~40 s)

```
python -m src.visualize --checkpoint experiments/stage3/last.pth
start experiments\figures
```

### Step 11 — Regenerate all report figures from the results  (~20 s)

```
python -m src.visualize_all
start experiments\figures_paper
```

### Step 12 — Training curves in TensorBoard

```
tensorboard --logdir experiments
```

Open **http://localhost:6006** in the browser. Press **Ctrl + C** in the terminal to stop.

---

## Part D — Backup: training (only if asked)

Training writes checkpoints. **Always keep `--exp_dir experiments/demo_...`** so the trained
models in `experiments\stage1..3` are never overwritten. Press **Ctrl + C** to stop at any time.

### Step 13 — Live training demo: KAIST Stage 1, one epoch  (~3 min)

```
python -m src.train --config configs/kaist.yaml --stage 1 --epochs 1 --exp_dir experiments/demo_stage1
```

Shows the training loop running; the loss after one epoch is ≈ 6.7 (the final model reached 0.56).

### Step 14 — Continue fine-tuning from the trained model, one epoch  (~8–10 min)

```
python -m src.train --config configs/kaist.yaml --stage 3 --epochs 1 --resume experiments/stage2/last.pth --exp_dir experiments/demo_stage3
```

### Step 15 — Evaluate the demo model (to show the pipeline end to end)

```
python -m src.evaluate --config configs/kaist.yaml --checkpoint experiments/demo_stage1/last.pth --split test
```

A 1-epoch model scores much lower than the final model; that is expected.

### Full training from scratch (for reference — takes many hours)

These are the exact commands used to produce the trained models. Do **not** run them during
the presentation: without `--exp_dir` they overwrite `experiments\`.

```
:: KAIST — 3 stages (~10 + 4 + 36 epochs used for the final model)
python -m src.train --config configs/kaist.yaml --stage 1
python -m src.train --config configs/kaist.yaml --stage 2 --resume experiments/stage1/last.pth
python -m src.train --config configs/kaist.yaml --stage 3 --resume experiments/stage2/last.pth

:: KITTI — 2 stages
python -m src.train_kitti --config configs/kitti.yaml --stage 1
python -m src.train_kitti --config configs/kitti.yaml --stage 2 --resume experiments/kitti_stage1/last.pth

:: FLIR — 3 stages
python -m src.train_flir --config configs/flir.yaml --stage 1
python -m src.train_flir --config configs/flir.yaml --stage 2 --resume experiments/flir_stage1/last.pth
python -m src.train_flir --config configs/flir.yaml --stage 3 --resume experiments/flir_stage2/last.pth

:: Baselines (RGB Only, Early Fusion, Late Fusion; 5 epochs each)
:: Note: this overwrites experiments\baselines\*.pth
python -m src.experiments.baseline_experiment
```

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `'conda' is not recognized` | You are in normal PowerShell. Open **Anaconda Prompt** instead (Step 0). |
| `No module named src` | You are in the wrong folder. Run `cd /d D:\aftmsf`. |
| `CUDA out of memory` | Close other programs using the GPU (games, browsers with video) and run the command again. |
| A command is slow | Plug in the charger; set Windows power mode to **Best performance**. |
| Stop a running command | Press **Ctrl + C**. |

## Folder map

| Folder / file | Contents |
|---|---|
| `experiments\stage3\last.pth` | Final KAIST model (all main results) |
| `experiments\kitti_stage2\last.pth` | KITTI model |
| `experiments\flir_stage3\last.pth` | FLIR model |
| `experiments\baselines\` | RGB Only, Early Fusion, Late Fusion models |
| `experiments\*.json`, `*_eval.yaml` | Saved results |
| `experiments\figures_paper\` | Report figures |
| `paper\` | Final report and presentation |

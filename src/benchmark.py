"""
Model statistics: parameter counts, inference speed and learned values.

FPS is measured on random 256x320 inputs, batch size 1, FP32, model forward
pass only (no data loading or post-processing), after 30 warm-up passes:
the median of 5 timed runs of 200 passes each. Results depend on GPU power
mode, so run it on mains power with nothing else using the GPU.

Writes experiments/model_stats.json (read by src.visualize_all).

Usage:
    python -m src.benchmark
"""

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.aftmsf_model import AFTMSFModel
from src.models.baselines import RGBOnlyModel, EarlyFusionModel, LateFusionModel


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def breakdown(model):
    d = {f'encoder_{k}': n_params(v) for k, v in model.encoders.items()}
    d.update(fault_detector=n_params(model.fault_detector),
             acaf=n_params(model.acaf),
             det_head=n_params(model.det_head),
             seg_head=n_params(model.seg_head),
             total=n_params(model))
    return d


def measure_fps(model, batch, device, n=200, warmup=30, trials=5):
    """Median FPS over `trials` timed runs of `n` forward passes each."""
    model.eval().to(device)
    runs = []
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                model(batch)
            torch.cuda.synchronize()
            runs.append(n / (time.perf_counter() - t0))
    model.cpu()
    torch.cuda.empty_cache()
    return round(sorted(runs)[len(runs) // 2], 2)


def main():
    device = torch.device('cuda')
    H, W   = 256, 320
    rgb    = torch.randn(1, 3, H, W, device=device)
    thr    = torch.randn(1, 1, H, W, device=device)
    dep    = torch.rand(1, 1, H, W, device=device)

    kaist = AFTMSFModel(['rgb', 'thermal'], 1, 2, 256, 4, 0.15, False)
    kitti = AFTMSFModel(['rgb', 'depth'],   2, 3, 256, 4, 0.15, False)
    flir  = AFTMSFModel(['rgb', 'thermal'], 3, 4, 256, 4, 0.15, False)
    baselines = {'RGBOnly': RGBOnlyModel(), 'EarlyFusion': EarlyFusionModel(),
                 'LateFusion': LateFusionModel()}

    stats = {'params': {'kaist': breakdown(kaist), 'kitti': breakdown(kitti),
                        'flir': breakdown(flir)}}
    stats['params'].update({k: n_params(m) for k, m in baselines.items()})

    rt = {'rgb': rgb, 'thermal': thr}
    stats['fps'] = {
        'AFTMSF_KAIST': measure_fps(kaist, rt, device),
        'AFTMSF_KITTI': measure_fps(kitti, {'rgb': rgb, 'depth': dep}, device),
        'AFTMSF_FLIR' : measure_fps(flir, rt, device),
    }
    for k, m in baselines.items():
        stats['fps'][k] = measure_fps(m, rt, device)

    # Learned values of the trained KAIST model
    ckpt = torch.load('experiments/stage3/last.pth', map_location='cpu',
                      weights_only=False)
    kaist.load_state_dict(ckpt['model'])
    tau = torch.exp(kaist.acaf.scale_fusion['0'].log_temperature).clamp(0.1, 10.0)
    stats['learned'] = {
        'temperature': tau.item(),
        'signal_weights': {m: F.softmax(e.w, 0).tolist()
                           for m, e in kaist.fault_detector.estimators.items()},
    }

    # Fusion weights implied by the measured confidences of each fault condition
    fault_path = Path('experiments/fault_experiment_results.json')
    if fault_path.exists():
        faults = json.load(open(fault_path))
        stats['fusion_weights'] = {}
        for k, v in faults.items():
            w = torch.softmax(torch.tensor([v['rgb_conf'], v['thermal_conf']]) / tau, 0)
            stats['fusion_weights'][k] = [round(w[0].item(), 4), round(w[1].item(), 4)]

    print(json.dumps(stats, indent=1))
    out = Path('experiments/model_stats.json')
    with open(out, 'w') as f:
        json.dump(stats, f, indent=1)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()

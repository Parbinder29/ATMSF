"""
FLIR Evaluation Script for AFTMSF System
Usage:
    python -m src.evaluate_flir
           --checkpoint experiments/flir_stage3/last.pth
"""

import argparse
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.models.aftmsf_model import AFTMSFModel
from src.datasets.flir_dataset import FLIRDataset, flir_collate_fn
from src.evaluate import MAPMetric, MIoUMetric, decode_fcos_predictions


def evaluate_flir(config_path: str, checkpoint: str,
                  split: str = 'val'):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Split      : {split}")

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
    print(f"Loaded epoch {ckpt['epoch']}, "
          f"loss={ckpt['loss']:.4f}")

    flir_root = r'D:\aftmsf\data\flir\FLIR_ADAS_1_3'
    dataset   = FLIRDataset(
        root=flir_root,
        split=split,
        img_h=cfg['dataset']['img_height'],
        img_w=cfg['dataset']['img_width'],
        fault_inject=False,
    )
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=flir_collate_fn,
    )
    print(f"Evaluating {len(dataset)} samples...")

    map_m  = MAPMetric(
        num_classes=cfg['dataset']['num_classes'],
        iou_thrs=[0.5]
    )
    miou_m = MIoUMetric(
        num_classes=cfg['dataset']['num_classes'] + 1
    )

    total_time = 0.0
    n_images   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            rgb     = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            H, W    = rgb.shape[2], rgb.shape[3]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()

            output = model(
                {'rgb': rgb, 'thermal': thermal},
                target_size=None,
            )

            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.time() - t0
            n_images   += rgb.shape[0]

            preds = decode_fcos_predictions(
                output['det_preds'], H, W,
                score_thr=0.1, nms_thr=0.5,
            )
            map_m.update(preds, batch['boxes'], H, W)

            seg_ds = F.interpolate(
                batch['seg_mask'].unsqueeze(1).float().to(device),
                size=(H//4, W//4), mode='nearest',
            ).squeeze(1).long()
            miou_m.update(output['seg_logits'], seg_ds)

    map_r  = map_m.compute()
    miou_r = miou_m.compute()
    fps    = n_images / total_time

    print("\n" + "="*50)
    print("FLIR EVALUATION RESULTS")
    print("="*50)
    for k, v in map_r.items():
        print(f"  {k:20s}: {v*100:.2f}%")
    for k, v in miou_r.items():
        print(f"  {k:20s}: {v*100:.2f}%")
    print(f"  FPS                 : {fps:.1f}")
    print("="*50)

    import yaml as _yaml
    results = {
        'map' : {k: float(v) for k,v in map_r.items()},
        'miou': {k: float(v) for k,v in miou_r.items()},
        'fps' : fps,
    }
    with open('experiments/flir_eval.yaml', 'w') as f:
        _yaml.dump(results, f)
    print("Saved to experiments/flir_eval.yaml")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/flir.yaml')
    parser.add_argument('--checkpoint',
                        default='experiments/flir_stage3/last.pth')
    parser.add_argument('--split', default='val')
    args = parser.parse_args() 
    evaluate_flir(args.config, args.checkpoint, args.split)
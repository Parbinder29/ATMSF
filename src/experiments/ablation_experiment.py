"""
Fault Detector Ablation for AFTMSF (KAIST test set)

Inference-time ablations of the trained stage-3 model (no retraining):
    full              : trained model as-is
    no_entropy        : entropy signal removed from the confidence score
                        (remaining two signals renormalised)
    no_fault_detector : every modality gets the same confidence (0.5),
                        so ACAF fusion weights are equal

The clean-data rows are repeated under every fault condition with and
without the fault detector, using the same seeded faults as
fault_experiment.py.

Usage:
    python -m src.experiments.ablation_experiment
           --checkpoint experiments/stage3/last.pth
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.aftmsf_model import AFTMSFModel
from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn
from src.experiments.fault_experiment import run_experiment


FAULTS = [('rgb', 'dropout'), ('rgb', 'blur'), ('rgb', 'occlusion'),
          ('rgb', 'gaussian_noise'), ('thermal', 'dropout'),
          ('thermal', 'blur'), ('thermal', 'occlusion'),
          ('thermal', 'gaussian_noise')]


def main(config_path, checkpoint, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AFTMSFModel(['rgb', 'thermal'], 1, 2, 256, 4, 0.15, False)
    ckpt  = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)

    loader = DataLoader(
        KAISTDataset(config_path, split='test', fault_inject=False),
        batch_size=4, shuffle=False, num_workers=0,
        collate_fn=kaist_collate_fn,
    )

    estimators  = model.fault_detector.estimators
    original_fd = model.fault_detector.forward

    def constant_confidence(features):
        B = next(iter(features.values()))['0'].shape[0]
        conf  = {m: torch.full((B,), 0.5, device=device) for m in features}
        flags = {m: torch.zeros(B, dtype=torch.bool, device=device)
                 for m in features}
        return conf, flags, {}

    def evaluate(mod=None, mode=None):
        torch.manual_seed(seed)
        r = run_experiment(model, loader, mod, mode, device, 1)
        return {k: round(float(v), 4) for k, v in r.items()}

    results = {'clean': {}, 'faults': {}}

    # Clean data
    results['clean']['full'] = evaluate()

    saved = {m: e.w.data.clone() for m, e in estimators.items()}
    for e in estimators.values():
        e.w.data[2] = -1e4          # softmax weight of entropy -> 0
    results['clean']['no_entropy'] = evaluate()
    for m, e in estimators.items():
        e.w.data.copy_(saved[m])

    model.fault_detector.forward = constant_confidence
    results['clean']['no_fault_detector'] = evaluate()
    model.fault_detector.forward = original_fd

    for name, r in results['clean'].items():
        print(f"clean {name:20s} mAP={r['mAP@0.5']:.2f}%  mIoU={r['mIoU']:.2f}%")

    # Every fault, with and without the fault detector
    for mod, mode in FAULTS:
        row = {}
        for label, fd in [('with_fd', original_fd),
                          ('without_fd', constant_confidence)]:
            model.fault_detector.forward = fd
            row[label] = evaluate(mod, mode)
        model.fault_detector.forward = original_fd
        results['faults'][f'{mod}_{mode}'] = row
        print(f"{mod}_{mode:15s} with FD {row['with_fd']['mAP@0.5']:.2f}%  "
              f"without FD {row['without_fd']['mAP@0.5']:.2f}%")

    out = Path('experiments/fault_detector_ablation.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/kaist.yaml')
    parser.add_argument('--checkpoint', default='experiments/stage3/last.pth')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.seed)

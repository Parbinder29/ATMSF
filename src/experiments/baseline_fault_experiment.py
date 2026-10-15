"""
Baseline Robustness Experiment (KAIST test set)

Evaluates the trained RGB Only, Early Fusion and Late Fusion baselines
under the same seeded faults as fault_experiment.py, so their behaviour
under sensor failure can be compared with AFTMSF.

Usage:
    python -m src.experiments.baseline_fault_experiment
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.baselines import RGBOnlyModel, EarlyFusionModel, LateFusionModel
from src.datasets.kaist_dataset import KAISTDataset, kaist_collate_fn
from src.experiments.baseline_experiment import evaluate_baseline
from src.experiments.fault_experiment import apply_fault


class FaultyBaseline(torch.nn.Module):
    """Wraps a baseline so one modality is corrupted before the forward pass."""

    def __init__(self, model, modality=None, mode=None):
        super().__init__()
        self.model, self.modality, self.mode = model, modality, mode

    def forward(self, batch, target_size=None):
        if self.modality is not None:
            batch = dict(batch)
            batch[self.modality] = apply_fault(batch[self.modality], self.mode)
        return self.model(batch, target_size=target_size)


CONDITIONS = [(None, None), ('rgb', 'dropout'), ('rgb', 'blur'),
              ('rgb', 'occlusion'), ('rgb', 'gaussian_noise'),
              ('thermal', 'dropout'), ('thermal', 'blur'),
              ('thermal', 'occlusion'), ('thermal', 'gaussian_noise')]


def main(config_path, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(
        KAISTDataset(config_path, split='test', fault_inject=False),
        batch_size=4, shuffle=False, num_workers=0,
        collate_fn=kaist_collate_fn,
    )

    results = {}
    for name, cls in [('RGBOnly', RGBOnlyModel),
                      ('EarlyFusion', EarlyFusionModel),
                      ('LateFusion', LateFusionModel)]:
        model = cls(1, 2)
        model.load_state_dict(torch.load(
            f'experiments/baselines/{name}.pth', map_location='cpu'))
        model.eval().to(device)
        results[name] = {}

        for mod, mode in CONDITIONS:
            if name == 'RGBOnly' and mod == 'thermal':
                continue                       # no thermal input
            key = 'clean' if mod is None else f'{mod}_{mode}'
            torch.manual_seed(seed)
            r = evaluate_baseline(FaultyBaseline(model, mod, mode),
                                  loader, device, 1)
            results[name][key] = {'mAP': round(float(r['mAP@0.5']), 2),
                                  'mIoU': round(float(r['mIoU']), 2)}
            print(f"{name:12s} {key:24s} mAP={results[name][key]['mAP']:.2f}%")

        model.cpu()
        torch.cuda.empty_cache()

    out = Path('experiments/baseline_fault_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/kaist.yaml')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.config, args.seed)

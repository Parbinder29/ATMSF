import torch
import yaml
from torch.utils.data import DataLoader
from src.models.aftmsf_model import AFTMSFModel
from src.datasets.flir_dataset import FLIRDataset, flir_collate_fn
from src.evaluate import decode_fcos_predictions

device = torch.device('cuda')
cfg    = yaml.safe_load(open('configs/flir.yaml'))

model = AFTMSFModel(
    modalities=['rgb','thermal'],
    num_det_classes=cfg['dataset']['num_classes'],
    num_seg_classes=cfg['dataset']['num_classes']+1,
    fpn_channels=256, num_heads=4,
    fault_thresh=0.15, grad_checkpoint=False,
)
ckpt = torch.load('experiments/flir_stage1/last.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval().to(device)

ds     = FLIRDataset(split='val', fault_inject=False)
loader = DataLoader(ds, batch_size=2,
                    collate_fn=flir_collate_fn,
                    num_workers=0)
batch  = next(iter(loader))

with torch.no_grad():
    out = model(
        {'rgb': batch['rgb'].to(device),
         'thermal': batch['thermal'].to(device)},
        target_size=None
    )

# Check raw prediction scores
import torch.nn.functional as F
for pred in out['det_preds']:
    scores = torch.sigmoid(pred['cls'])
    print(f"Scale {pred['scale']}: "
          f"max={scores.max():.4f} "
          f"mean={scores.mean():.4f} "
          f">0.1: {(scores>0.1).sum().item()} "
          f">0.3: {(scores>0.3).sum().item()}")

# Try with very low threshold
preds = decode_fcos_predictions(
    out['det_preds'], 256, 320,
    score_thr=0.01, nms_thr=0.5
)
print(f"\nPredictions with thr=0.01: {[len(p) for p in preds]}")
print(f"GT boxes: {[len(b) for b in batch['boxes']]}")
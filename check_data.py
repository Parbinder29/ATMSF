import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from src.models.aftmsf_model import AFTMSFModel
from src.datasets.flir_dataset import FLIRDataset, flir_collate_fn
from src.evaluate import decode_fcos_predictions, box_iou

device = torch.device('cuda')
cfg    = yaml.safe_load(open('configs/flir.yaml'))

model = AFTMSFModel(
    modalities=['rgb','thermal'],
    num_det_classes=cfg['dataset']['num_classes'],
    num_seg_classes=cfg['dataset']['num_classes']+1,
    fpn_channels=256, num_heads=4,
    fault_thresh=0.15, grad_checkpoint=False,
)
ckpt = torch.load('experiments/flir_stage2/last.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval().to(device)

ds     = FLIRDataset(split='val', fault_inject=False,
                     max_samples=20)
loader = DataLoader(ds, batch_size=1,
                    collate_fn=flir_collate_fn,
                    num_workers=0)

print("Checking prediction vs GT box overlap...")
total_gt = 0
total_matched = 0

for i, batch in enumerate(loader):
    if i >= 5:
        break

    with torch.no_grad():
        out = model(
            {'rgb': batch['rgb'].to(device),
             'thermal': batch['thermal'].to(device)},
            target_size=None
        )

    H, W = batch['rgb'].shape[2], batch['rgb'].shape[3]
    preds = decode_fcos_predictions(
        out['det_preds'], H, W,
        score_thr=0.05, nms_thr=0.5
    )

    gt_boxes  = batch['boxes'][0]   # (N,4) YOLO format
    pred_boxes = preds[0]           # (M,6)

    print(f"\nImage {i}: GT={len(gt_boxes)} boxes, "
          f"Pred={len(pred_boxes)} boxes")

    if len(gt_boxes) > 0:
        # Convert GT to xyxy absolute
        cx=gt_boxes[:,0]*W; cy=gt_boxes[:,1]*H
        bw=gt_boxes[:,2]*W; bh=gt_boxes[:,3]*H
        gt_xyxy = torch.stack([cx-bw/2,cy-bh/2,
                               cx+bw/2,cy+bh/2],dim=1)
        print(f"  GT boxes (xyxy): {gt_xyxy[:3].numpy()}")

    if len(pred_boxes) > 0:
        print(f"  Top pred scores: "
              f"{pred_boxes[:3,4].cpu().numpy()}")
        print(f"  Top pred boxes: "
              f"{pred_boxes[:3,:4].cpu().numpy()}")

        if len(gt_boxes) > 0:
            iou = box_iou(pred_boxes[:5,:4].cpu(),
                         gt_xyxy[:5])
            print(f"  Max IoU pred vs GT: {iou.max():.3f}")
            total_matched += (iou.max(dim=1).values > 0.5).sum().item()

    total_gt += len(gt_boxes)

print(f"\nSummary: {total_matched}/{total_gt} GT boxes matched")
print(f"Issue diagnosis:")
print(f"  If Max IoU > 0.5 exists → metric computation bug")
print(f"  If Max IoU always < 0.5 → model not detecting correctly")
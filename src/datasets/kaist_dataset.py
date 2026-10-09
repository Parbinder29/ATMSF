"""
KAIST Dataset Loader — YOLO Early Fusion Format
Handles RGBA images where:
    Channels 0,1,2 = RGB (visible)
    Channel 3      = Thermal (infrared / lwir)

Dataset structure:
    data/kaist/kaist_yolo_early/kaist_yolo_early/
        images/train/*.png   (RGBA, 640x512)
        images/val/*.png
        images/test/*.png
        labels/train/*.txt   (YOLO format: class cx cy w h)
        labels/val/*.txt
        labels/test/*.txt

Paper note:
    This early-fusion KAIST variant provides aligned RGB+Thermal
    pairs in a single file, ensuring perfect spatial alignment
    between modalities — ideal for evaluating fusion approaches.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


class KAISTDataset(Dataset):
    """
    KAIST Multispectral Pedestrian Dataset.
    RGBA early-fusion format — RGB + Thermal in single image.

    Args:
        config_path  : path to configs/kaist.yaml
        split        : 'train', 'val', or 'test'
        fault_inject : enable fault injection during training
    """

    BASE = Path(r'D:\aftmsf\data\kaist\kaist_yolo_early\kaist_yolo_early')

    def __init__(
        self,
        config_path  : str,
        split        : str  = 'train',
        fault_inject : bool = True,
    ):
        super().__init__()
        assert split in ('train', 'val', 'test')

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.split        = split
        self.fault_inject = fault_inject and (split == 'train')
        self.fault_cfg    = self.cfg['fault_injection']

        self.img_dir = self.BASE / 'images' / split
        self.lbl_dir = self.BASE / 'labels' / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images not found: {self.img_dir}")

        self.samples = sorted([
            f.stem for f in self.img_dir.glob('*.png')
        ])

        print(f"[KAISTDataset] {split}: {len(self.samples)} samples")

        self.h = self.cfg['dataset']['img_height']   # 512
        self.w = self.cfg['dataset']['img_width']    # 640

        # Normalisation stats
        self.mean_rgb     = self.cfg['preprocessing']['mean_rgb']
        self.std_rgb      = self.cfg['preprocessing']['std_rgb']
        self.mean_thermal = self.cfg['preprocessing']['mean_thermal']
        self.std_thermal  = self.cfg['preprocessing']['std_thermal']

        self.transform = self._build_transform()

    # ------------------------------------------------------------------
    def _build_transform(self):
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.h, self.w),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, hue=0.05, p=0.4),
                A.GaussianBlur(blur_limit=(3,5), p=0.2),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,
                clip=True,
            ))
        else:
            return A.Compose([
                A.Resize(self.h, self.w),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,
                clip=True,
            ))

    # ------------------------------------------------------------------
    def _load_label(self, stem: str):
        """
        Load YOLO label file.
        Returns boxes (N,4) in [cx,cy,w,h] normalised
        and labels (N,) int.
        """
        lbl_path = self.lbl_dir / f'{stem}.txt'
        boxes, labels = [], []

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls  = int(parts[0])
                    cxcywh = [float(x) for x in parts[1:5]]

                    # Filter invalid boxes
                    if cxcywh[2] <= 0 or cxcywh[3] <= 0:
                        continue

                    boxes.append(cxcywh)
                    labels.append(cls)

        if boxes:
            return (np.array(boxes,  dtype=np.float32),
                    np.array(labels, dtype=np.int64))
        return (np.zeros((0,4), dtype=np.float32),
                np.zeros((0,),  dtype=np.int64))

    # ------------------------------------------------------------------
    def _inject_fault(
        self,
        img : np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """Apply synthetic fault to image array (H,W,C) uint8."""
        img = img.copy()

        if mode == 'dropout':
            img[:] = 0

        elif mode == 'gaussian_noise':
            std   = self.fault_cfg['noise_std']
            noise = np.random.normal(0, std*255,
                                     img.shape).astype(np.int16)
            img   = np.clip(
                img.astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)

        elif mode == 'occlusion':
            h, w  = img.shape[:2]
            rh    = np.random.randint(int(0.2*h), int(0.5*h))
            rw    = np.random.randint(int(0.2*w), int(0.5*w))
            ry    = np.random.randint(0, h - rh)
            rx    = np.random.randint(0, w - rw)
            img[ry:ry+rh, rx:rx+rw] = 0

        elif mode == 'blur':
            k   = np.random.choice([15, 21, 31])
            img = cv2.GaussianBlur(img, (k,k), 0)

        return img

    def _get_fault_mode(self):
        if not self.fault_inject:
            return None
        if not self.fault_cfg['enabled']:
            return None
        if np.random.random() < self.fault_cfg['dropout_prob']:
            return np.random.choice(self.fault_cfg['modes'])
        return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        stem = self.samples[idx]

        # ── Load RGBA image ───────────────────────────────────────────
        img_path = self.img_dir / f'{stem}.png'
        rgba     = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if rgba is None:
            raise IOError(f"Cannot read: {img_path}")

        if rgba.ndim == 2:
            # Grayscale — expand
            rgba = np.stack([rgba]*4, axis=-1)
        elif rgba.shape[2] == 3:
            # No alpha — create zero thermal
            rgba = np.concatenate(
                [rgba, np.zeros((*rgba.shape[:2],1), dtype=np.uint8)],
                axis=-1
            )

        # Split channels
        # OpenCV loads as BGRA
        bgr     = rgba[:, :, :3]
        thermal = rgba[:, :,  3]                     # (H, W) uint8

        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (H,W,3)
        thr_3ch = np.stack([thermal]*3, axis=-1)     # (H,W,3) for albumentations

        # ── Load labels ───────────────────────────────────────────────
        boxes, labels = self._load_label(stem)
        boxes_list    = boxes.tolist()
        labels_list   = labels.tolist()

        # ── Fault injection ───────────────────────────────────────────
        rgb_fault_mode = self._get_fault_mode()
        thr_fault_mode = self._get_fault_mode()

        rgb_fault_flag = 0
        thr_fault_flag = 0

        if rgb_fault_mode:
            rgb            = self._inject_fault(rgb, rgb_fault_mode)
            rgb_fault_flag = 1
        if thr_fault_mode:
            thr_3ch        = self._inject_fault(thr_3ch, thr_fault_mode)
            thr_fault_flag = 1

        # ── Transforms ────────────────────────────────────────────────
        transformed = self.transform(
            image=rgb,
            bboxes=boxes_list,
            class_labels=labels_list,
        )
        rgb_aug     = transformed['image']          # (H,W,3) uint8
        boxes_aug   = transformed['bboxes']
        labels_aug  = transformed['class_labels']

        # Resize thermal to match
        thr_aug = cv2.resize(
            thr_3ch, (self.w, self.h),
            interpolation=cv2.INTER_LINEAR
        )

        # ── Normalise + convert to tensor ─────────────────────────────
        rgb_f   = rgb_aug.astype(np.float32) / 255.0
        thr_f   = thr_aug[:,:,0].astype(np.float32) / 255.0

        # Normalise RGB
        mean_r  = np.array(self.mean_rgb,     dtype=np.float32)
        std_r   = np.array(self.std_rgb,      dtype=np.float32)
        rgb_f   = (rgb_f - mean_r) / std_r

        # Normalise thermal
        thr_f   = (thr_f - self.mean_thermal[0]) / self.std_thermal[0]

        # To tensors
        rgb_t   = torch.from_numpy(
            rgb_f.transpose(2,0,1)
        ).float()                                    # (3,H,W)
        thr_t   = torch.from_numpy(
            thr_f[np.newaxis]
        ).float()                                    # (1,H,W)

        # Boxes and labels
        if boxes_aug:
            boxes_t  = torch.tensor(boxes_aug,  dtype=torch.float32)
            labels_t = torch.tensor(labels_aug, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0,4), dtype=torch.float32)
            labels_t = torch.zeros((0,),  dtype=torch.int64)

        # Segmentation mask from boxes (weak supervision)
        seg_mask = self._boxes_to_seg_mask(boxes_aug)

        return {
            'rgb'          : rgb_t,
            'thermal'      : thr_t,
            'boxes'        : boxes_t,
            'labels'       : labels_t,
            'seg_mask'     : seg_mask,
            'rgb_fault'    : torch.tensor(rgb_fault_flag,
                                          dtype=torch.float32),
            'thermal_fault': torch.tensor(thr_fault_flag,
                                          dtype=torch.float32),
            'meta'         : {'stem': stem},
        }

    # ------------------------------------------------------------------
    def _boxes_to_seg_mask(self, boxes: list) -> torch.Tensor:
        """
        Generate weak segmentation mask from bounding boxes.
        Inside box = class 1 (person), outside = class 0 (background).
        This is weak supervision — sufficient for initial training.
        """
        mask = torch.zeros(self.h, self.w, dtype=torch.long)

        for box in boxes:
            cx, cy, w, h = box
            x1 = int((cx - w/2) * self.w)
            y1 = int((cy - h/2) * self.h)
            x2 = int((cx + w/2) * self.w)
            y2 = int((cy + h/2) * self.h)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(self.w-1, x2); y2 = min(self.h-1, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1

        return mask

    def __len__(self):
        return len(self.samples)


# ------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------
def kaist_collate_fn(batch):
    return {
        'rgb'          : torch.stack([b['rgb']           for b in batch]),
        'thermal'      : torch.stack([b['thermal']       for b in batch]),
        'boxes'        : [b['boxes']                     for b in batch],
        'labels'       : [b['labels']                   for b in batch],
        'seg_mask'     : torch.stack([b['seg_mask']      for b in batch]),
        'rgb_fault'    : torch.stack([b['rgb_fault']     for b in batch]),
        'thermal_fault': torch.stack([b['thermal_fault'] for b in batch]),
        'meta'         : [b['meta']                      for b in batch],
    }


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == '__main__':
    ds = KAISTDataset('configs/kaist.yaml', split='train')
    print(f"Dataset size: {len(ds)}")

    item = ds[0]
    print(f"RGB shape    : {item['rgb'].shape}")
    print(f"Thermal shape: {item['thermal'].shape}")
    print(f"Boxes        : {item['boxes'].shape}")
    print(f"Labels       : {item['labels']}")
    print(f"Seg mask     : {item['seg_mask'].shape} "
          f"unique={item['seg_mask'].unique().tolist()}")
    print(f"RGB fault    : {item['rgb_fault'].item()}")
    print(f"Thermal fault: {item['thermal_fault'].item()}")

    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=4,
        collate_fn=kaist_collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    print(f"\nBatch RGB    : {batch['rgb'].shape}")
    print(f"Batch Thermal: {batch['thermal'].shape}")
    print(f"Batch seg    : {batch['seg_mask'].shape}")
    print("KAISTDataset OK")
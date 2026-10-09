"""
FLIR ADAS Thermal Dataset Loader for AFTMSF System
Dataset: FLIR ADAS v1.3

Structure:
    FLIR_ADAS_1_3/
        train/
            RGB/              *.jpg
            thermal_8_bit/    *.jpeg
            thermal_annotations.json
        val/
            RGB/
            thermal_8_bit/
            thermal_annotations.json

Annotations: COCO format JSON
Classes: 1=person, 2=car, 3=bicycle
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A

FLIR_CLASSES = {
    1: 0,   # person
    2: 1,   # bicycle
    3: 2,   # car
    17: 0,  # dog → treated as person class (rare, 226 samples)
}
FLIR_NAMES = {0: 'person', 1: 'bicycle', 2: 'car'}
FLIR_ROOT    = r'D:\aftmsf\data\flir\FLIR_ADAS_1_3'


class FLIRDataset(Dataset):
    """
    FLIR ADAS Thermal Dataset.
    RGB + Thermal image pairs with COCO bounding box annotations.

    Args:
        root         : path to FLIR_ADAS_1_3/
        split        : 'train' or 'val'
        img_h, img_w : target image size
        fault_inject : enable fault injection
        max_samples  : limit dataset size
    """

    def __init__(
        self,
        root        : str   = FLIR_ROOT,
        split       : str   = 'train',
        img_h       : int   = 256,
        img_w       : int   = 320,
        fault_inject: bool  = True,
        max_samples : int   = None,
    ):
        super().__init__()
        assert split in ('train', 'val')

        self.root         = Path(root)
        self.split        = split
        self.img_h        = img_h
        self.img_w        = img_w
        self.fault_inject = fault_inject and (split == 'train')

        self.split_dir   = self.root / split
        self.rgb_dir     = self.split_dir / 'RGB'
        self.thermal_dir = self.split_dir / 'thermal_8_bit'
        self.ann_path    = self.split_dir / 'thermal_annotations.json'

        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB dir not found: {self.rgb_dir}")

        # Load COCO annotations
        with open(self.ann_path) as f:
            self.coco = json.load(f)

        # Build image_id → annotations mapping
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Build sample list
        self.samples = self._build_sample_list()
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"[FLIRDataset] {split}: {len(self.samples)} samples")

        # Normalisation stats
        self.mean_rgb     = [0.4820, 0.4820, 0.4820]
        self.std_rgb      = [0.2350, 0.2350, 0.2350]
        self.mean_thermal = 0.4500
        self.std_thermal  = 0.2200

        # Fault injection
        self.fault_modes = ['dropout','gaussian_noise','occlusion','blur']
        self.fault_prob  = 0.15

        self.transform = self._build_transform()

    # ------------------------------------------------------------------
    def _build_sample_list(self):
        samples = []
        for img_info in self.coco['images']:
            img_id       = img_info['id']
            filename     = img_info['file_name']
            thermal_name = Path(filename).name
            thermal_path = self.thermal_dir / thermal_name

            if not thermal_path.exists():
                stem = Path(thermal_name).stem
                for ext in ['.jpeg', '.jpg', '.png']:
                    alt = self.thermal_dir / (stem + ext)
                    if alt.exists():
                        thermal_path = alt
                        break
                else:
                    continue

            stem     = Path(thermal_name).stem
            rgb_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = self.rgb_dir / (stem + ext)
                if candidate.exists():
                    rgb_path = candidate
                    break

            if rgb_path is None:
                continue

            samples.append({
                'img_id'      : img_id,
                'img_info'    : img_info,
                'rgb_path'    : rgb_path,
                'thermal_path': thermal_path,
            })
        return samples

    # ------------------------------------------------------------------
    def _load_annotations(self, img_id, orig_w, orig_h):
        anns           = self.img_to_anns.get(img_id, [])
        boxes, labels  = [], []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in FLIR_CLASSES:
                continue
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x+w, y+h
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(orig_w, x2); y2 = min(orig_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2-x1) < 3 or (y2-y1) < 3:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(FLIR_CLASSES[cat_id])

        if boxes:
            return (np.array(boxes,  dtype=np.float32),
                    np.array(labels, dtype=np.int64))
        return (np.zeros((0,4), dtype=np.float32),
                np.zeros((0,),  dtype=np.int64))

    # ------------------------------------------------------------------
    def _build_transform(self):
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.img_h, self.img_w),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.1, p=0.4),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3, clip=True,
            ))
        else:
            return A.Compose([
                A.Resize(self.img_h, self.img_w),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3, clip=True,
            ))

    # ------------------------------------------------------------------
    def _inject_fault(self, img, mode):
        img = img.copy()
        if mode == 'dropout':
            img[:] = 0
        elif mode == 'gaussian_noise':
            noise = np.random.normal(0, 0.05*255,
                                     img.shape).astype(np.int16)
            img   = np.clip(img.astype(np.int16)+noise,
                            0, 255).astype(np.uint8)
        elif mode == 'occlusion':
            h, w  = img.shape[:2]
            rh    = np.random.randint(int(0.2*h), int(0.5*h))
            rw    = np.random.randint(int(0.2*w), int(0.5*w))
            ry    = np.random.randint(0, h-rh)
            rx    = np.random.randint(0, w-rw)
            img[ry:ry+rh, rx:rx+rw] = 0
        elif mode == 'blur':
            k   = np.random.choice([15, 21, 31])
            img = cv2.GaussianBlur(img, (k,k), 0)
        return img

    def _get_fault_mode(self):
        if not self.fault_inject:
            return None
        if np.random.random() < self.fault_prob:
            return np.random.choice(self.fault_modes)
        return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB
        rgb = cv2.imread(str(sample['rgb_path']))
        if rgb is None:
            rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        # Load Thermal
        thr = cv2.imread(str(sample['thermal_path']),
                         cv2.IMREAD_GRAYSCALE)
        if thr is None:
            thr = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        # Load annotations
        boxes, labels = self._load_annotations(
            sample['img_id'], orig_w, orig_h
        )

        # Transform
        transformed = self.transform(
            image=rgb,
            bboxes=boxes.tolist(),
            class_labels=labels.tolist(),
        )
        rgb_aug    = transformed['image']
        boxes_aug  = transformed['bboxes']
        labels_aug = transformed['class_labels']

        # Resize thermal
        thr_aug = cv2.resize(thr, (self.img_w, self.img_h),
                             interpolation=cv2.INTER_LINEAR)

        # Fault injection
        rgb_fault_mode = self._get_fault_mode()
        thr_fault_mode = self._get_fault_mode()
        rgb_fault_flag = 0
        thr_fault_flag = 0

        if rgb_fault_mode:
            rgb_aug        = self._inject_fault(rgb_aug, rgb_fault_mode)
            rgb_fault_flag = 1
        if thr_fault_mode:
            thr_3ch        = np.stack([thr_aug]*3, -1)
            thr_aug        = self._inject_fault(
                thr_3ch, thr_fault_mode
            )[:,:,0]
            thr_fault_flag = 1

        # Normalise RGB
        rgb_f  = rgb_aug.astype(np.float32) / 255.0
        mean_r = np.array(self.mean_rgb, dtype=np.float32)
        std_r  = np.array(self.std_rgb,  dtype=np.float32)
        rgb_f  = (rgb_f - mean_r) / std_r

        # Normalise thermal
        thr_f  = thr_aug.astype(np.float32) / 255.0
        thr_f  = (thr_f - self.mean_thermal) / self.std_thermal

        # To tensors
        rgb_t = torch.from_numpy(
            rgb_f.transpose(2,0,1)).float()
        thr_t = torch.from_numpy(
            thr_f[np.newaxis]).float()

        # Boxes to YOLO format
        if boxes_aug:
            ba   = np.array(boxes_aug, dtype=np.float32)
            cx   = ((ba[:,0]+ba[:,2])/2) / self.img_w
            cy   = ((ba[:,1]+ba[:,3])/2) / self.img_h
            bw   = (ba[:,2]-ba[:,0])     / self.img_w
            bh   = (ba[:,3]-ba[:,1])     / self.img_h
            boxes_t  = torch.from_numpy(
                np.stack([cx,cy,bw,bh],axis=1)).float()
            labels_t = torch.tensor(labels_aug, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0,4), dtype=torch.float32)
            labels_t = torch.zeros((0,),  dtype=torch.int64)

        # Segmentation mask
        seg_mask = torch.zeros(self.img_h, self.img_w, dtype=torch.long)
        for box in boxes_aug:
            x1,y1,x2,y2 = [int(v) for v in box]
            x1=max(0,x1); y1=max(0,y1)
            x2=min(self.img_w-1,x2); y2=min(self.img_h-1,y2)
            if x2>x1 and y2>y1:
                seg_mask[y1:y2, x1:x2] = 1

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
            'meta'         : {'img_id': sample['img_id'],
                             'split': self.split},
        }

    def __len__(self):
        return len(self.samples)


# ------------------------------------------------------------------
def flir_collate_fn(batch):
    return {
        'rgb'          : torch.stack([b['rgb']       for b in batch]),
        'thermal'      : torch.stack([b['thermal']   for b in batch]),
        'boxes'        : [b['boxes']                 for b in batch],
        'labels'       : [b['labels']                for b in batch],
        'seg_mask'     : torch.stack([b['seg_mask']  for b in batch]),
        'rgb_fault'    : torch.stack([b['rgb_fault'] for b in batch]),
        'thermal_fault': torch.stack([b['thermal_fault']
                                      for b in batch]),
        'meta'         : [b['meta']                  for b in batch],
    }


# ------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing FLIRDataset...")
    ds = FLIRDataset(split='train', fault_inject=False)
    print(f"Dataset size: {len(ds)}")

    item = ds[0]
    print(f"RGB shape    : {item['rgb'].shape}")
    print(f"Thermal shape: {item['thermal'].shape}")
    print(f"Boxes        : {item['boxes'].shape}")
    print(f"Labels       : {item['labels']}")
    print(f"Seg mask     : {item['seg_mask'].unique().tolist()}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4,
                        collate_fn=flir_collate_fn,
                        num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch RGB    : {batch['rgb'].shape}")
    print(f"Batch Thermal: {batch['thermal'].shape}")
    print(f"Batch seg    : {batch['seg_mask'].shape}")
    print("FLIRDataset OK")
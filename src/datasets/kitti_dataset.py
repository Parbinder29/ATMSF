"""
KITTI Dataset Loader for AFTMSF System
Handles RGB images + LiDAR point clouds (projected to depth maps)

Dataset structure:
    data/kitti/training/
        image_2/    *.png  — Left RGB images (1242x375)
        label_2/    *.txt  — KITTI format annotations
        velodyne/   *.bin  — LiDAR point clouds (x,y,z,intensity)

KITTI label format (per line):
    class truncated occluded alpha x1 y1 x2 y2 h w l X Y Z rotation

Classes we use:
    Pedestrian, Cyclist → class 0 (person)
    Car, Van, Truck     → class 1 (vehicle)

LiDAR processing:
    1. Load raw point cloud (N,4) — x,y,z,intensity
    2. Project to image plane using calibration
    3. Create sparse depth map (H,W) — publishable contribution

For paper: cite KITTI as second dataset showing cross-dataset
generalisation of the AFTMSF framework.
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


# ── KITTI class mapping ────────────────────────────────────────────────
KITTI_CLASSES = {
    'Pedestrian' : 0,
    'Person_sitting': 0,
    'Cyclist'    : 0,
    'Car'        : 1,
    'Van'        : 1,
    'Truck'      : 1,
}

# Ignore these classes
KITTI_IGNORE = {
    'DontCare', 'Misc', 'Tram'
}


class KITTIDataset(Dataset):
    """
    KITTI Object Detection Dataset with LiDAR depth.

    Modalities:
        - RGB image (3, H, W)
        - Projected LiDAR depth map (1, H, W) — sparse

    Args:
        root         : path to D:/aftmsf/data/kitti/
        split        : 'train' or 'val' (we split training set)
        img_h, img_w : target image size
        val_fraction : fraction of training data for validation
        fault_inject : enable fault injection
    """

    ROOT = Path(r'D:\aftmsf\data\kitti')

    def __init__(
        self,
        root         : str   = None,
        split        : str   = 'train',
        img_h        : int   = 256,
        img_w        : int   = 320,
        val_fraction : float = 0.1,
        fault_inject : bool  = True,
        max_samples: int = None,
    ):
        super().__init__()
        assert split in ('train', 'val', 'test')

        self.root         = Path(root) if root else self.ROOT
        self.split        = split
        self.img_h        = img_h
        self.img_w        = img_w
        self.fault_inject = fault_inject and (split == 'train')

        self.train_dir = self.root / 'training'
        self.img_dir   = self.train_dir / 'image_2'
        self.lbl_dir   = self.train_dir / 'label_2'
        self.velo_dir  = self.train_dir / 'velodyne'

        # Get all sample indices
        all_ids = sorted([
            f.stem for f in self.img_dir.glob('*.png')
        ])
        n_val   = max(1, int(len(all_ids) * val_fraction))

        if split == 'train':
            self.sample_ids = all_ids[n_val:]
        elif split == 'val':
            # Always use training val split — has labels
            self.sample_ids = all_ids[:n_val]
            self.img_dir    = self.train_dir / 'image_2'
            self.velo_dir   = self.train_dir / 'velodyne'
            self.lbl_dir    = self.train_dir / 'label_2'
        else:
            self.sample_ids = all_ids[:n_val]

        # Limit samples for faster training
        if max_samples and len(self.sample_ids) > max_samples:
            self.sample_ids = self.sample_ids[:max_samples]
        ''''
        if split == 'train':
            self.sample_ids = all_ids[n_val:]
        elif split == 'val':
            self.sample_ids = all_ids[:n_val]

        # Limit samples for faster training
        if max_samples and len(self.sample_ids) > max_samples:
            self.sample_ids = self.sample_ids[:max_samples]
        else:
            # test — use testing folder if available
            test_dir = self.root / 'testing' / 'image_2'
            if test_dir.exists():
                self.img_dir    = self.root / 'testing' / 'image_2'
                self.velo_dir   = self.root / 'testing' / 'velodyne'
                self.lbl_dir    = None
                self.sample_ids = sorted([
                    f.stem for f in self.img_dir.glob('*.png')
                ])
            else:
                self.sample_ids = all_ids[:n_val]
'''
        print(f"[KITTIDataset] {split}: "
              f"{len(self.sample_ids)} samples")

        # Normalisation stats (ImageNet for RGB)
        self.mean_rgb = [0.3450, 0.3690, 0.3780]
        self.std_rgb  = [0.2770, 0.2840, 0.2890]
        # Depth normalisation (KITTI max depth ~80m)
        self.max_depth = 80.0

        # Fault injection config
        self.fault_modes   = ['dropout','gaussian_noise',
                              'occlusion','blur']
        self.fault_prob    = 0.15

        self.transform = self._build_transform()

    # ------------------------------------------------------------------
    def _build_transform(self):
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.img_h, self.img_w),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, p=0.4),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3,
                clip=True,
            ))
        else:
            return A.Compose([
                A.Resize(self.img_h, self.img_w),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3,
                clip=True,
            ))

    # ------------------------------------------------------------------
    def _load_label(self, sample_id: str):
        """
        Parse KITTI label file.
        Returns boxes (N,4) in [x1,y1,x2,y2] pixel coords
        and labels (N,) int.
        """
        if self.lbl_dir is None:
            return (np.zeros((0,4), dtype=np.float32),
                    np.zeros((0,),  dtype=np.int64))

        lbl_path = self.lbl_dir / f'{sample_id}.txt'
        boxes, labels = [], []

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue
                    cls_name = parts[0]
                    if cls_name in KITTI_IGNORE:
                        continue
                    if cls_name not in KITTI_CLASSES:
                        continue

                    x1 = float(parts[4])
                    y1 = float(parts[5])
                    x2 = float(parts[6])
                    y2 = float(parts[7])

                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2-x1) < 5 or (y2-y1) < 5:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    labels.append(KITTI_CLASSES[cls_name])

        if boxes:
            return (np.array(boxes,  dtype=np.float32),
                    np.array(labels, dtype=np.int64))
        return (np.zeros((0,4), dtype=np.float32),
                np.zeros((0,),  dtype=np.int64))

    # ------------------------------------------------------------------
    def _load_depth(self, sample_id: str,
                    img_h: int, img_w: int) -> np.ndarray:
        """
        Load LiDAR point cloud and project to depth map.

        Steps:
            1. Load binary point cloud (N,4) — x,y,z,intensity
            2. Filter points in front of camera (x > 0)
            3. Simple projection using approximate KITTI calibration
            4. Create sparse depth image (H,W) normalised to [0,1]

        Returns: (H, W) float32 depth map, values in [0,1]
                 0 = no LiDAR return, >0 = depth normalised by max_depth
        """
        velo_path = self.velo_dir / f'{sample_id}.bin'

        if not velo_path.exists():
            return np.zeros((img_h, img_w), dtype=np.float32)

        # Load point cloud — (N, 4) x,y,z,intensity
        pts = np.fromfile(str(velo_path),
                          dtype=np.float32).reshape(-1, 4)

        # Keep only forward-facing points (x > 0)
        pts = pts[pts[:, 0] > 0]

        if len(pts) == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)

        # Approximate KITTI projection to image plane
        # Using average KITTI camera intrinsics
        fx = 721.5377
        fy = 721.5377
        cx = 609.5593
        cy = 172.8540

        # Original KITTI image size
        orig_h = 375
        orig_w = 1242

        # Project: u = fx * y / x + cx (KITTI velodyne coord)
        # In KITTI: x=forward, y=left, z=up
        # Camera: X=right, Y=down, Z=forward
        # Transformation: Xc = -y, Yc = -z, Zc = x
        Xc = -pts[:, 1]
        Yc = -pts[:, 2]
        Zc =  pts[:, 0]

        # Filter points behind camera
        mask = Zc > 0.5
        Xc, Yc, Zc = Xc[mask], Yc[mask], Zc[mask]

        # Project to pixel coordinates
        u = (fx * Xc / Zc + cx).astype(np.int32)
        v = (fy * Yc / Zc + cy).astype(np.int32)

        # Filter within image bounds
        valid = (u >= 0) & (u < orig_w) & \
                (v >= 0) & (v < orig_h)
        u, v, depth = u[valid], v[valid], Zc[valid]

        # Create depth map at original resolution
        depth_map = np.zeros((orig_h, orig_w),
                             dtype=np.float32)
        depth_norm = np.clip(depth / self.max_depth, 0, 1)
        depth_map[v, u] = depth_norm

        # Resize to target resolution
        depth_map = cv2.resize(
            depth_map, (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        )

        return depth_map

    # ------------------------------------------------------------------
    def _inject_fault(self, img: np.ndarray,
                      mode: str) -> np.ndarray:
        img = img.copy()
        if mode == 'dropout':
            img[:] = 0
        elif mode == 'gaussian_noise':
            noise = np.random.normal(0, 0.05*255,
                                     img.shape).astype(np.int16)
            img   = np.clip(img.astype(np.int16)+noise,
                           0,255).astype(np.uint8)
        elif mode == 'occlusion':
            h,w   = img.shape[:2]
            rh    = np.random.randint(int(0.2*h),int(0.5*h))
            rw    = np.random.randint(int(0.2*w),int(0.5*w))
            ry    = np.random.randint(0, h-rh)
            rx    = np.random.randint(0, w-rw)
            img[ry:ry+rh, rx:rx+rw] = 0
        elif mode == 'blur':
            k   = np.random.choice([15,21,31])
            img = cv2.GaussianBlur(img,(k,k),0)
        return img

    def _get_fault_mode(self):
        if not self.fault_inject:
            return None
        if np.random.random() < self.fault_prob:
            return np.random.choice(self.fault_modes)
        return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]

        # ── Load RGB ──────────────────────────────────────────────────
        img_path = self.img_dir / f'{sample_id}.png'
        rgb      = cv2.imread(str(img_path))
        if rgb is None:
            raise IOError(f"Cannot read: {img_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = rgb.shape[:2]

        # ── Load labels ───────────────────────────────────────────────
        boxes, labels = self._load_label(sample_id)
        boxes_list    = boxes.tolist()
        labels_list   = labels.tolist()

        # ── Apply transform ───────────────────────────────────────────
        transformed = self.transform(
            image=rgb,
            bboxes=boxes_list,
            class_labels=labels_list,
        )
        rgb_aug     = transformed['image']
        boxes_aug   = transformed['bboxes']
        labels_aug  = transformed['class_labels']

        # ── Load depth map ────────────────────────────────────────────
        depth = self._load_depth(
            sample_id, self.img_h, self.img_w
        )   # (H, W) float32 in [0,1]

        # ── Fault injection ───────────────────────────────────────────
        rgb_fault_mode   = self._get_fault_mode()
        depth_fault_mode = self._get_fault_mode()
        rgb_fault_flag   = 0
        depth_fault_flag = 0

        if rgb_fault_mode:
            rgb_aug          = self._inject_fault(
                rgb_aug, rgb_fault_mode
            )
            rgb_fault_flag   = 1

        if depth_fault_mode:
            depth_uint8      = (depth*255).astype(np.uint8)
            depth_3ch        = np.stack([depth_uint8]*3, -1)
            depth_3ch        = self._inject_fault(
                depth_3ch, depth_fault_mode
            )
            depth            = depth_3ch[:,:,0].astype(
                np.float32
            ) / 255.0
            depth_fault_flag = 1

        # ── Normalise RGB ─────────────────────────────────────────────
        rgb_f  = rgb_aug.astype(np.float32) / 255.0
        mean_r = np.array(self.mean_rgb, dtype=np.float32)
        std_r  = np.array(self.std_rgb,  dtype=np.float32)
        rgb_f  = (rgb_f - mean_r) / std_r

        # ── To tensors ────────────────────────────────────────────────
        rgb_t   = torch.from_numpy(
            rgb_f.transpose(2,0,1)
        ).float()                           # (3,H,W)
        depth_t = torch.from_numpy(
            depth[np.newaxis]
        ).float()                           # (1,H,W)

        # ── Boxes to YOLO format for compatibility ────────────────────
        if boxes_aug:
            ba = np.array(boxes_aug, dtype=np.float32)
            # Convert xyxy → cxcywh normalised
            cx = ((ba[:,0]+ba[:,2])/2) / self.img_w
            cy = ((ba[:,1]+ba[:,3])/2) / self.img_h
            bw = (ba[:,2]-ba[:,0])     / self.img_w
            bh = (ba[:,3]-ba[:,1])     / self.img_h
            boxes_yolo = np.stack([cx,cy,bw,bh],axis=1)
            boxes_t    = torch.from_numpy(
                boxes_yolo
            ).float()
            labels_t   = torch.tensor(labels_aug,
                                      dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0,4), dtype=torch.float32)
            labels_t = torch.zeros((0,),  dtype=torch.int64)

        # ── Segmentation mask from boxes ──────────────────────────────
        seg_mask = torch.zeros(
            self.img_h, self.img_w, dtype=torch.long
        )
        for box in boxes_aug:
            x1,y1,x2,y2 = [int(v) for v in box]
            x1=max(0,x1); y1=max(0,y1)
            x2=min(self.img_w-1,x2)
            y2=min(self.img_h-1,y2)
            if x2>x1 and y2>y1:
                seg_mask[y1:y2, x1:x2] = 1

        return {
            'rgb'         : rgb_t,
            'depth'       : depth_t,
            'boxes'       : boxes_t,
            'labels'      : labels_t,
            'seg_mask'    : seg_mask,
            'rgb_fault'   : torch.tensor(
                rgb_fault_flag, dtype=torch.float32
            ),
            'depth_fault' : torch.tensor(
                depth_fault_flag, dtype=torch.float32
            ),
            'meta'        : {'sample_id': sample_id},
        }

    def __len__(self):
        return len(self.sample_ids)


# ------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------
def kitti_collate_fn(batch):
    return {
        'rgb'        : torch.stack([b['rgb']       for b in batch]),
        'depth'      : torch.stack([b['depth']     for b in batch]),
        'boxes'      : [b['boxes']                 for b in batch],
        'labels'     : [b['labels']                for b in batch],
        'seg_mask'   : torch.stack([b['seg_mask']  for b in batch]),
        'rgb_fault'  : torch.stack([b['rgb_fault'] for b in batch]),
        'depth_fault': torch.stack([b['depth_fault']
                                    for b in batch]),
        'meta'       : [b['meta']                  for b in batch],
    }


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == '__main__':
    ds = KITTIDataset(split='train', img_h=256, img_w=320)
    print(f"Dataset size: {len(ds)}")

    item = ds[0]
    print(f"RGB shape   : {item['rgb'].shape}")
    print(f"Depth shape : {item['depth'].shape}")
    print(f"Depth stats : min={item['depth'].min():.3f} "
          f"max={item['depth'].max():.3f} "
          f"nonzero={( item['depth']>0).float().mean():.3f}")
    print(f"Boxes       : {item['boxes'].shape}")
    print(f"Labels      : {item['labels']}")
    print(f"Seg mask    : {item['seg_mask'].unique().tolist()}")

    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=4,
        collate_fn=kitti_collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    print(f"\nBatch RGB  : {batch['rgb'].shape}")
    print(f"Batch Depth: {batch['depth'].shape}")
    print(f"Batch seg  : {batch['seg_mask'].shape}")
    print("KITTIDataset OK")
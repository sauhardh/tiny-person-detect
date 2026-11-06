import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import augment_dataset
from .targets import build_targets

# ImageNet normalisation constants (channel order: R, G, B)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TileDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        root: Optional[str] = None,
        split: Optional[str] = "train",
        source: Optional[str] = None,
        train: bool = True,
        T: int = 640,
        stride: int = 4,
        out: int = 160,
    ):
        manifest_csv = Path(manifest_csv)
        if root is None:
            # CSV lives at  <project>/data/meta/all_tiles.csv
            # CSV paths are "data/nomad_tiles/..."  -> root = <project>/
            root = manifest_csv.parent.parent.parent
        self.root = Path(root)
        self.train = train
        self.T = T
        self.stride = stride
        self.out = out

        # ------------------------------------------------------------------ #
        # Load & filter the manifest                                          #
        # ------------------------------------------------------------------ #
        self.samples: list = []

        with open(manifest_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is not None and row["split"] != split:
                    continue
                if source is not None and row["source"] != source:
                    continue

                img_path = self.root / row["image_path"]
                lbl_path = self.root / row["label_path"]
                self.samples.append((img_path, lbl_path))

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {manifest_csv} for "
                f"split={split!r}, source={source!r}."
            )

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _load_boxes(self, lbl_path: Path) -> np.ndarray:
        boxes = []

        if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # parts[0] is class id (always 0 — "person")
                cx, cy, w, h = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )

                x1 = (cx - w / 2.0) * self.T
                y1 = (cy - h / 2.0) * self.T
                x2 = (cx + w / 2.0) * self.T
                y2 = (cy + h / 2.0) * self.T

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(boxes, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        # 1. Load image (H x W x 3, uint8) and bounding boxes (N x 4, pixels)
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        boxes = self._load_boxes(lbl_path)

        # 2. On-the-fly augmentation (train only)
        if self.train:
            img, boxes = augment_dataset(img, boxes, self.T)

        # 3. Normalise: float32 in [0, 1], then subtract ImageNet mean / std
        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - _MEAN) / _STD  # (H, W, 3)
        img_f = np.ascontiguousarray(img_f.transpose(2, 0, 1))  # -> (3, H, W)

        # 4. Build CenterNet target maps from (possibly augmented) pixel boxes
        hm, wh, off, mask = build_targets(boxes, self.out, self.stride)

        return {
            "image": torch.from_numpy(img_f),  # (3, 640, 640)
            "hm": torch.from_numpy(hm),  # (1, 160, 160)
            "wh": torch.from_numpy(wh),  # (2, 160, 160)
            "off": torch.from_numpy(off),  # (2, 160, 160)
            "mask": torch.from_numpy(mask),  # (1, 160, 160)
        }

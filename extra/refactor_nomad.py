import os
import re
import csv
import glob
import shutil
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


class RefactorNomad:
    """
    Refactors the nested NOMAD-YOLO dataset into a standardized flat YOLO dataset:
    - train & val: sliced into 640x640 tiles with recalculated bounding boxes
    - test: flattened at FULL resolution (5472x3078) with original bounding boxes for sliding-window evaluation
    """

    def __init__(
        self,
        src_dir: Optional[str] = None,
        out_dir: Optional[str] = None,
        tile_size: int = 640,
        n_person_tiles: int = 2,
        n_bg_tiles: int = 4,
        min_keep: float = 0.5,
        seed: int = 42,
    ):
        # Auto-detect source directory regardless of current working directory
        if src_dir is None:
            candidate_a = SCRIPT_DIR / "data" / "NOMAD-YOLO"
            candidate_b = Path.cwd() / "dataset" / "data" / "NOMAD-YOLO"
            candidate_c = Path.cwd() / "data" / "NOMAD-YOLO"

            if candidate_a.exists():
                self.src_dir = candidate_a
            elif candidate_b.exists():
                self.src_dir = candidate_b
            elif candidate_c.exists():
                self.src_dir = candidate_c
            else:
                self.src_dir = candidate_a
        else:
            p = Path(src_dir)
            if not p.is_absolute() and not p.exists():
                if (SCRIPT_DIR / p).exists():
                    p = SCRIPT_DIR / p
            self.src_dir = p.resolve()

        if out_dir is None:
            self.out_dir = (SCRIPT_DIR / "data" / "nomad_tiles").resolve()
        else:
            p = Path(out_dir)
            if not p.is_absolute() and not p.exists() and (SCRIPT_DIR / p).parent.exists():
                p = SCRIPT_DIR / p
            self.out_dir = p.resolve()

        self.tile_size = tile_size
        self.n_person_tiles = n_person_tiles
        self.n_bg_tiles = n_bg_tiles
        self.min_keep = min_keep
        self.seed = seed

    def create_output_dirs(self, splits: Tuple[str, ...] = ("train", "val", "test")) -> None:
        """Create clean target directory structure for images, labels, and metadata."""
        for split in splits:
            (self.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (self.out_dir / "meta").mkdir(parents=True, exist_ok=True)

    def read_yolo_labels(self, label_path: str, img_w: int, img_h: int) -> np.ndarray:
        """
        Reads YOLO format label file (class cx cy w h normalized)
        and converts to absolute coordinates: [x1, y1, x2, y2, class_id].
        """
        boxes = []
        if os.path.isfile(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w / 2.0) * img_w
                        y1 = (cy - h / 2.0) * img_h
                        x2 = (cx + w / 2.0) * img_w
                        y2 = (cy + h / 2.0) * img_h
                        boxes.append([x1, y1, x2, y2, cls_id])

        if not boxes:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(boxes, dtype=np.float32)

    def get_person_tile_origin(
        self, box: np.ndarray, img_w: int, img_h: int, rng: random.Random
    ) -> Tuple[int, int]:
        """Computes top-left (tx, ty) of a tile ensuring the person box is inside."""
        x1, y1, x2, y2 = box[:4]
        T = self.tile_size

        lo_x = max(0, int(np.ceil(x2)) - T)
        hi_x = min(int(x1), img_w - T)
        lo_y = max(0, int(np.ceil(y2)) - T)
        hi_y = min(int(y1), img_h - T)

        if lo_x > hi_x:
            lo_x = hi_x = int(np.clip((x1 + x2) / 2.0 - T / 2.0, 0, img_w - T))
        if lo_y > hi_y:
            lo_y = hi_y = int(np.clip((y1 + y2) / 2.0 - T / 2.0, 0, img_h - T))

        tx = rng.randint(lo_x, hi_x)
        ty = rng.randint(lo_y, hi_y)
        return tx, ty

    def get_bg_tile_origin(
        self, boxes: np.ndarray, img_w: int, img_h: int, rng: random.Random, max_tries: int = 50
    ) -> Optional[Tuple[int, int]]:
        """Finds a top-left (tx, ty) for a tile that does not overlap with any person box."""
        T = self.tile_size
        for _ in range(max_tries):
            tx = rng.randint(0, img_w - T)
            ty = rng.randint(0, img_h - T)

            has_overlap = False
            for b in boxes:
                bx1, by1, bx2, by2 = b[:4]
                if not (bx2 <= tx or bx1 >= tx + T or by2 <= ty or by1 >= ty + T):
                    has_overlap = True
                    break

            if not has_overlap:
                return tx, ty
        return None

    def update_labels(
        self, boxes: np.ndarray, tx: int, ty: int
    ) -> List[Tuple[int, float, float, float, float]]:
        """
        Clips boxes to tile coordinates and returns normalized YOLO annotations:
        [(class_id, cx, cy, w, h), ...] relative to (tile_size, tile_size).
        Applies the 50% area overlap rule.
        """
        T = self.tile_size
        updated = []

        for b in boxes:
            x1, y1, x2, y2, cls_id = b
            orig_area = max(0.0, (x2 - x1) * (y2 - y1))
            if orig_area <= 0:
                continue

            ix1 = max(x1, tx)
            iy1 = max(y1, ty)
            ix2 = min(x2, tx + T)
            iy2 = min(y2, ty + T)

            iw = ix2 - ix1
            ih = iy2 - iy1

            if iw <= 0 or ih <= 0:
                continue

            inter_area = iw * ih
            if (inter_area / orig_area) < self.min_keep:
                continue

            local_x1 = ix1 - tx
            local_y1 = iy1 - ty
            local_x2 = ix2 - tx
            local_y2 = iy2 - ty

            norm_cx = ((local_x1 + local_x2) / 2.0) / T
            norm_cy = ((local_y1 + local_y2) / 2.0) / T
            norm_w = (local_x2 - local_x1) / T
            norm_h = (local_y2 - local_y1) / T

            updated.append((int(cls_id), norm_cx, norm_cy, norm_w, norm_h))

        return updated

    def tile_image(self, img: Image.Image, tx: int, ty: int) -> Image.Image:
        """Crops a (tile_size x tile_size) patch from the image at origin (tx, ty)."""
        return img.crop((tx, ty, tx + self.tile_size, ty + self.tile_size))

    def save_tile(
        self,
        tile_img: Image.Image,
        updated_labels: List[Tuple[int, float, float, float, float]],
        tile_name: str,
        split: str,
    ) -> Tuple[str, str]:
        """Saves tile image (.jpg) and label (.txt) to the corresponding split directory."""
        img_out_path = self.out_dir / "images" / split / f"{tile_name}.jpg"
        lbl_out_path = self.out_dir / "labels" / split / f"{tile_name}.txt"

        tile_img.save(img_out_path, quality=95)

        with open(lbl_out_path, "w", encoding="utf-8") as f:
            for cls_id, cx, cy, w, h in updated_labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        return str(img_out_path), str(lbl_out_path)

    def process_frame(
        self, img_path: str, split: str
    ) -> List[Dict[str, Any]]:
        """Processes a single high-resolution NOMAD frame into person and background tiles."""
        rng = random.Random(f"{self.seed}_{img_path}")

        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Error opening image {img_path}]: {e}")
            return []

        w_full, h_full = im.size
        T = self.tile_size

        if w_full < T or h_full < T:
            return []

        lbl_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        boxes = self.read_yolo_labels(lbl_path, w_full, h_full)

        stem = Path(img_path).stem
        actor_match = re.search(r"Actor\d+", img_path)
        dist_match = re.search(r"_a(\d+)", stem)

        actor = actor_match.group(0) if actor_match else "unknown"
        dist_m = int(dist_match.group(1)) if dist_match else -1

        rows = []

        def emit_tile(k: int, kind: str, tx: int, ty: int) -> None:
            tile = self.tile_image(im, tx, ty)
            updated_boxes = self.update_labels(boxes, tx, ty)
            tile_name = f"{stem}_{kind}{k}"
            self.save_tile(tile, updated_boxes, tile_name, split)

            rows.append({
                "tile": tile_name,
                "split": split,
                "actor": actor,
                "dist_m": dist_m,
                "tx": tx,
                "ty": ty,
                "n_boxes": len(updated_boxes),
                "is_bg": (kind == "bg"),
            })

        # 1. Person tiles
        for k in range(self.n_person_tiles):
            if len(boxes) > 0:
                selected_box = boxes[rng.randrange(len(boxes))]
                tx, ty = self.get_person_tile_origin(selected_box, w_full, h_full, rng)
                emit_tile(k, "p", tx, ty)

        # 2. Background tiles
        for k in range(self.n_bg_tiles):
            bg_origin = self.get_bg_tile_origin(boxes, w_full, h_full, rng)
            if bg_origin is not None:
                tx, ty = bg_origin
                emit_tile(k, "bg", tx, ty)

        return rows

    def refactor_test(self, use_symlinks: bool = True) -> int:
        """
        Refactors ONLY the 'test' split without tiling:
        - Flattens nested ActorXXX/ActorXXX_aYY/ images into <out_dir>/images/test/
        - Flattens corresponding label files into <out_dir>/labels/test/
        - Preserves 100% full original resolution (5472x3078) and original bounding boxes.
        - Uses symlinks by default (0 extra disk space) or copies if requested.
        """
        test_img_dir = self.out_dir / "images" / "test"
        test_lbl_dir = self.out_dir / "labels" / "test"
        # Clean existing test split to avoid mixing tiled and full-frame
        if test_img_dir.exists():
            shutil.rmtree(test_img_dir)
        if test_lbl_dir.exists():
            shutil.rmtree(test_lbl_dir)
        test_img_dir.mkdir(parents=True, exist_ok=True)
        test_lbl_dir.mkdir(parents=True, exist_ok=True)

        pattern = str(self.src_dir / "images" / "test" / "**" / "*.jpg")
        test_files = sorted(glob.glob(pattern, recursive=True))

        print("=" * 70)
        print("Refactoring TEST Split (Full Resolution / No Tiling)")
        print(f"Source:      {self.src_dir / 'images' / 'test'}")
        print(f"Destination: {test_img_dir}")
        print(f"Total Test Images: {len(test_files)}")
        print(f"Mode:        {'Symlink (0 MB disk)' if use_symlinks else 'Copy'}")
        print("=" * 70)

        if not test_files:
            print(f"[Warning] No test images found matching: {pattern}")
            return 0

        count = 0
        test_metadata = []

        for src_img_str in tqdm(test_files, desc="test (full-frame)"):
            src_img = Path(src_img_str).resolve()
            stem = src_img.stem

            dest_img = test_img_dir / f"{stem}.jpg"
            dest_lbl = test_lbl_dir / f"{stem}.txt"

            # Mirror path to find label file
            src_lbl = Path(str(src_img).replace("/images/", "/labels/")).with_suffix(".txt")

            # Remove previous file or broken symlink if exists
            if dest_img.is_symlink() or dest_img.exists():
                dest_img.unlink()
            if dest_lbl.is_symlink() or dest_lbl.exists():
                dest_lbl.unlink()

            # Handle Image
            if use_symlinks:
                dest_img.symlink_to(src_img)
            else:
                shutil.copy2(src_img, dest_img)

            # Handle Label
            if src_lbl.exists():
                if use_symlinks:
                    dest_lbl.symlink_to(src_lbl)
                else:
                    shutil.copy2(src_lbl, dest_lbl)
            else:
                dest_lbl.touch()

            # Metadata recording
            actor_match = re.search(r"Actor\d+", str(src_img))
            dist_match = re.search(r"_a(\d+)", stem)
            test_metadata.append({
                "image": f"{stem}.jpg",
                "split": "test",
                "actor": actor_match.group(0) if actor_match else "unknown",
                "dist_m": int(dist_match.group(1)) if dist_match else -1,
                "is_tiled": False,
                "full_resolution": True,
            })
            count += 1

        # Write test metadata
        meta_csv = self.out_dir / "meta" / "test_frames.csv"
        (self.out_dir / "meta").mkdir(parents=True, exist_ok=True)
        if test_metadata:
            fieldnames = ["image", "split", "actor", "dist_m", "is_tiled", "full_resolution"]
            with open(meta_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(test_metadata)
            print(f"Saved test metadata to: {meta_csv}")

        print(f"Successfully refactored {count} full-resolution test frames into {test_img_dir}")
        return count

    def run(
        self,
        splits: Tuple[str, ...] = ("train", "val"),
        max_workers: int = 4,
        flatten_test: bool = True,
        use_symlinks: bool = True,
    ) -> None:
        """
        Runs the complete refactoring pipeline:
        - Tiles the training/validation splits (e.g. train, val)
        - Flattens the test split at full resolution (without tiling)
        - Writes metadata CSV and data.yaml
        """
        print("=" * 70)
        print("Starting NOMAD Dataset Refactoring")
        print(f"Source:      {self.src_dir}")
        print(f"Destination: {self.out_dir}")
        print(f"Tile Size:   {self.tile_size}x{self.tile_size}")
        print(f"Tiles/frame: {self.n_person_tiles} person + {self.n_bg_tiles} background")
        print(f"Tiled Splits: {splits}")
        print(f"Flatten Test: {flatten_test} (Full resolution)")
        print("=" * 70)

        if not self.src_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found at: {self.src_dir}\n"
                f"Please verify where NOMAD-YOLO is located."
            )

        self.create_output_dirs(splits)
        all_metadata = []

        # 1. Process Tiled Splits (train, val)
        for split in splits:
            pattern = str(self.src_dir / "images" / split / "**" / "*.jpg")
            files = sorted(glob.glob(pattern, recursive=True))

            print(f"\nProcessing split '{split}': found {len(files)} full-frame images...")
            if not files:
                print(f"  [Warning] No .jpg files found matching: {pattern}")
                continue

            split_rows = []
            if max_workers > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_worker_process_frame, self, f, split): f
                        for f in files
                    }
                    for fut in tqdm(as_completed(futures), total=len(files), desc=split):
                        split_rows.extend(fut.result())
            else:
                for f in tqdm(files, desc=split):
                    split_rows.extend(self.process_frame(f, split))

            print(f"Generated {len(split_rows)} tiles for split '{split}'")
            all_metadata.extend(split_rows)

        # 2. Flatten Test Split without tiling
        if flatten_test:
            self.refactor_test(use_symlinks=use_symlinks)

        # Save metadata CSV for tiles
        meta_csv = self.out_dir / "meta" / "tiles.csv"
        if all_metadata:
            fieldnames = ["tile", "split", "actor", "dist_m", "tx", "ty", "n_boxes", "is_bg"]
            with open(meta_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_metadata)
            print(f"\nSaved tile metadata CSV to: {meta_csv}")

        # Save YOLO data.yaml
        yaml_content = f"""# NOMAD Dataset (Tiles for train/val, Full-frame for test)
path: {self.out_dir}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['human']
"""
        yaml_path = self.out_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"Saved YOLO configuration to: {yaml_path}")
        print("\nRefactoring complete! All datasets are ready for training.")


def _worker_process_frame(refactorer: RefactorNomad, img_path: str, split: str):
    """Helper module-level function for multiprocessing pickling."""
    return refactorer.process_frame(img_path, split)


def refactor_test_dataset(src_dir: Optional[str] = None, out_dir: Optional[str] = None, use_symlinks: bool = True) -> int:
    """Convenience standalone function to refactor test split only."""
    refactorer = RefactorNomad(src_dir=src_dir, out_dir=out_dir)
    return refactorer.refactor_test(use_symlinks=use_symlinks)


def main():
    parser = argparse.ArgumentParser(description="Refactor NOMAD-YOLO dataset")
    parser.add_argument("--src", type=str, default=None, help="Source NOMAD-YOLO path")
    parser.add_argument("--out", type=str, default=None, help="Output nomad_tiles path")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile width & height")
    parser.add_argument("--person-tiles", type=int, default=2, help="Number of person tiles per frame")
    parser.add_argument("--bg-tiles", type=int, default=2, help="Number of background tiles per frame")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to tile (default: train val)")
    parser.add_argument("--test-only", action="store_true", help="Only refactor test split (no tiling)")
    parser.add_argument("--copy-test", action="store_true", help="Copy test files instead of symlink")
    args = parser.parse_args()

    refactorer = RefactorNomad(
        src_dir=args.src,
        out_dir=args.out,
        tile_size=args.tile_size,
        n_person_tiles=args.person_tiles,
        n_bg_tiles=args.bg_tiles,
    )

    if args.test_only:
        refactorer.refactor_test(use_symlinks=(not args.copy_test))
    else:
        refactorer.run(
            splits=tuple(args.splits),
            max_workers=args.workers,
            flatten_test=True,
            use_symlinks=(not args.copy_test),
        )


if __name__ == "__main__":
    main()

"""
src/infer/run_test.py — Full-Frame Tiled Evaluation against Test Split
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

from src.detector import Detector
from src.infer.tiled import predict_tiled
from src.eval.metrics import evaluate


def load_ground_truths(nomad_root: str, split: str = "test") -> Dict[str, Any]:
    """Loads full-frame test images and their un-tiled ground truth annotations."""
    gts = {}
    img_files = sorted(glob.glob(f"{nomad_root}/images/{split}/**/*.jpg", recursive=True))

    print(f"Loading {len(img_files)} ground truth images for split '{split}'...")
    for p in img_files:
        with Image.open(p) as im:
            W, H = im.size

        lab_p = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        boxes = []
        if os.path.exists(lab_p):
            with open(lab_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = (float(x) for x in parts[1:5])
                        boxes.append([
                            (cx - w / 2.0) * W,
                            (cy - h / 2.0) * H,
                            (cx + w / 2.0) * W,
                            (cy + h / 2.0) * H,
                        ])

        # Extract distance altitude from filename (e.g., Actor001_a30_f0012 -> 30)
        dist_match = re.search(r"_a(\d+)", p)
        dist = int(dist_match.group(1)) if dist_match else 0

        gts[p] = {
            "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
            "dist": [dist] * len(boxes),
        }

    return gts


def run_evaluation(
    weights_path: str,
    nomad_root: str = "data/NOMAD-YOLO",
    conf_thr: float = 0.25,
    nms_iou: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dev = torch.device(device)
    print(f"Loading Detector with weights: {weights_path} onto {dev}...")
    model = Detector("resnet18", pretrained=False).to(dev)
    state = torch.load(weights_path, map_location=dev)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    gts = load_ground_truths(nomad_root, split="test")

    predictions = {}
    print(f"Running full-frame tiled inference over {len(gts)} frames...")
    for idx, (img_path, _) in enumerate(gts.items(), 1):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        boxes, scores = predict_tiled(
            model=model,
            img=img_np,
            T=640,
            overlap=0.25,
            thr=0.05,
            nms_iou=nms_iou,
            device=dev,
        )
        predictions[img_path] = (boxes, scores)
        if idx % 25 == 0 or idx == len(gts):
            print(f"  Processed [{idx:>3d}/{len(gts)}] frames")

    print("\nComputing Precision, Recall, mAP@50 and Distance-Stratified Metrics...")
    results = evaluate(predictions, gts, conf_thr=conf_thr, iou_thr=0.5)

    print("\n" + "=" * 60)
    print("                     EVALUATION RESULTS                    ")
    print("=" * 60)
    print(f"  AP@50 (mAP50)       : {results['ap50']:.4f}")
    print(f"  Precision (@{conf_thr:.2f})  : {results['precision']:.4f}")
    print(f"  Recall (@{conf_thr:.2f})     : {results['recall']:.4f}")
    print(f"  FP per image        : {results['fp_per_image']:.4f}")
    print(f"  Total Ground Truth  : {results['total_gt']}")
    print(f"  True Positives (TP) : {results['tp']}")
    print(f"  False Positives (FP): {results['fp']}")
    print("-" * 60)
    print("  Recall by Flight Distance (Altitude):")
    for dist, stats in results["recall_by_dist"].items():
        print(f"    {dist:>2d}m : {stats['found']:>2d}/{stats['total']:>2d} found ({stats['recall'] * 100:.1f}%)")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt or checkpoint")
    parser.add_argument("--nomad-root", type=str, default="data/NOMAD-YOLO")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()

    run_evaluation(
        weights_path=args.weights,
        nomad_root=args.nomad_root,
        conf_thr=args.conf,
        nms_iou=args.nms_iou,
    )

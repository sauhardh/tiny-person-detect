"""
Visualization script for verifying the CenterNet target generation pipeline.

Inspects random tiles from the dataset and plots:
  1. Original Tile Image with Ground Truth Bounding Box(es)
  2. Generated Gaussian Heatmap (160x160)
  3. Overlay of Heatmap + Heatmap Centers (derived from mask / peaks) on original image

Can be run as a standalone script or imported:
    python -m pre.visualization --samples 24 --save-dir ./visualizations
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from .data import TileDataset, _MEAN, _STD


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts (3, H, W) normalized float tensor back to (H, W, 3) uint8 image.
    """
    img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * _STD) + _MEAN
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


def visualize_batch(
    dataset: TileDataset,
    num_samples: int = 24,
    save_path: Optional[str] = None,
    seed: int = 42,
    stride: int = 4,
):
    """
    Renders a grid of samples showing:
      Column 1: Image + GT Boxes
      Column 2: Heatmap
      Column 3: Heatmap Overlay + Center dots

    Args:
        dataset: Instantiated TileDataset
        num_samples: Number of samples to render (default: 24)
        save_path: Path to save figure, or None to display interactively
        seed: Random seed for sample selection
        stride: CenterNet downsampling factor (default: 4)
    """
    rng = random.Random(seed)
    total_len = len(dataset)
    num_samples = min(num_samples, total_len)

    # Prioritize non-empty (person) samples so the visualization is informative
    indices = []
    # Collect candidate indices
    all_indices = list(range(total_len))
    rng.shuffle(all_indices)

    # First pass: find samples with at least one person
    for idx in all_indices:
        _, lbl_path = dataset.samples[idx]
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            indices.append(idx)
        if len(indices) >= int(num_samples * 0.75):
            break

    # Fill remainder with random samples (which may include background tiles)
    for idx in all_indices:
        if idx not in indices:
            indices.append(idx)
        if len(indices) >= num_samples:
            break

    rng.shuffle(indices)

    # 3 columns per sample: Image+BBox, Heatmap, Overlay+Center
    cols = 3
    fig, axes = plt.subplots(num_samples, cols, figsize=(12, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, idx in enumerate(indices):
        data = dataset[idx]
        img_np = denormalize_image(data["image"])
        hm = data["hm"][0].cpu().numpy()       # (160, 160)
        mask = data["mask"][0].cpu().numpy()   # (160, 160)
        wh = data["wh"].cpu().numpy()          # (2, 160, 160)
        off = data["off"].cpu().numpy()        # (2, 160, 160)

        img_path, lbl_path = dataset.samples[idx]
        title_prefix = f"#{idx} ({img_path.stem[:18]})"

        # -------------------------------------------------------------
        # Col 1: Original Image with Bounding Boxes (reconstructed from targets)
        # -------------------------------------------------------------
        ax1 = axes[row_idx, 0]
        ax1.imshow(img_np)
        ax1.set_title(f"{title_prefix}\nImage + GT Boxes", fontsize=10)
        ax1.axis("off")

        # Find positive centers from mask
        ys, xs = np.where(mask > 0.5)
        num_persons = len(xs)

        for cy_cell, cx_cell in zip(ys, xs):
            w_box = wh[0, cy_cell, cx_cell] * stride
            h_box = wh[1, cy_cell, cx_cell] * stride
            off_x = off[0, cy_cell, cx_cell]
            off_y = off[1, cy_cell, cx_cell]

            cx_px = (cx_cell + off_x) * stride
            cy_px = (cy_cell + off_y) * stride

            x1 = cx_px - w_box / 2.0
            y1 = cy_px - h_box / 2.0

            rect = patches.Rectangle(
                (x1, y1),
                w_box,
                h_box,
                linewidth=1.8,
                edgecolor="#00FF66",
                facecolor="none",
            )
            ax1.add_patch(rect)
            ax1.plot(cx_px, cy_px, "r+", markersize=8, markeredgewidth=2)

        # -------------------------------------------------------------
        # Col 2: Heatmap
        # -------------------------------------------------------------
        ax2 = axes[row_idx, 1]
        im2 = ax2.imshow(hm, cmap="magma", vmin=0.0, vmax=1.0)
        peak_val = hm.max()
        ax2.set_title(f"Heatmap (160x160)\nPeak: {peak_val:.2f} | People: {num_persons}", fontsize=10)
        ax2.axis("off")

        # -------------------------------------------------------------
        # Col 3: Heatmap Overlay + Heatmap Center Dots on Original Image
        # -------------------------------------------------------------
        ax3 = axes[row_idx, 2]
        ax3.imshow(img_np)
        # Overlay heatmap with alpha transparency
        ax3.imshow(hm, cmap="jet", alpha=0.45, extent=[0, 640, 640, 0])

        # Plot center peaks
        for cy_cell, cx_cell in zip(ys, xs):
            off_x = off[0, cy_cell, cx_cell]
            off_y = off[1, cy_cell, cx_cell]
            cx_px = (cx_cell + off_x) * stride
            cy_px = (cy_cell + off_y) * stride

            # Center circle
            circle = plt.Circle(
                (cx_px, cy_px),
                radius=10,
                color="#FFFF00",
                fill=False,
                linewidth=2.0,
            )
            ax3.add_patch(circle)
            ax3.plot(cx_px, cy_px, "y*", markersize=9)

        ax3.set_title(f"Heatmap Overlay\n+ Centers ({num_persons} detected)", fontsize=10)
        ax3.axis("off")

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Visualization saved successfully: {out_file.resolve()}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CenterNet tile targets (Image, BBox, Heatmap, Centers)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/meta/all_tiles.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["nomad", "sar", None],
        help="Dataset source filter ('nomad', 'sar', or None for mixed)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="Number of random samples to visualize (default: 24)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="visualizations",
        help="Output directory to save images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentations during visualization",
    )

    args = parser.parse_args()

    print(f"Loading TileDataset: split={args.split}, source={args.source}, train={args.augment}...")
    dataset = TileDataset(
        manifest_csv=args.manifest,
        split=args.split,
        source=args.source,
        train=args.augment,
    )
    print(f"Dataset loaded: {len(dataset)} total samples.")

    save_file = Path(args.save_dir) / f"targets_preview_{args.split}_{args.samples}samples.png"
    print(f"Generating visualization for {args.samples} samples -> {save_file}...")

    visualize_batch(
        dataset=dataset,
        num_samples=args.samples,
        save_path=str(save_file),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

# Convenience helper for interactive notebooks or scripts
def visualize_sample(dataset: TileDataset, index: int, save_path: Optional[str] = None):
    visualize_batch(dataset=dataset, num_samples=1, save_path=save_path, seed=index)

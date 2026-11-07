import math
from typing import Tuple
import numpy as np


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """
    Calculates the Gaussian radius from object height and width using
    the CenterNet formula.

    Ensures that two nearby objects will not have their Gaussian peaks merged
    if their overlap is below `min_overlap`.
    """
    a1, b1, c1 = 1.0, h + w, w * h * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1**2 - 4.0 * a1 * c1))
    r1 = (b1 - sq1) / (2.0 * a1)

    a2, b2, c2 = 4.0, 2.0 * (h + w), (1.0 - min_overlap) * w * h
    sq2 = math.sqrt(max(0.0, b2**2 - 4.0 * a2 * c2))
    r2 = (b2 - sq2) / (2.0 * a2)

    a3, b3, c3 = (
        4.0 * min_overlap,
        -2.0 * min_overlap * (h + w),
        (min_overlap - 1.0) * w * h,
    )
    sq3 = math.sqrt(max(0.0, b3**2 - 4.0 * a3 * c3))
    r3 = (b3 + sq3) / (2.0 * a3)

    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """
    Draws a 2D Gaussian bell-curve peak (max value 1.0) centered at (cx, cy).
    Modified in-place.
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2.0 * sigma * sigma))

    H, W = heatmap.shape
    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)

    patch = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    gaussian_patch = g[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(patch, gaussian_patch, out=patch)


def build_targets(
    boxes_xyxy: np.ndarray, out: int = 160, stride: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts Nx4 pixel bounding boxes [x1, y1, x2, y2] into CenterNet target maps.

    Args:
        boxes_xyxy: Nx4 NumPy array in pixel coordinates [0, T]
        out: spatial output grid size (160 for T=640 and stride=4)
        stride: downsampling factor of the detector (4)

    Returns:
        hm:   (1, 160, 160) - Gaussian heatmap with peaks of 1.0 at person centers
        wh:   (2, 160, 160) - Width and height in output grid cells
        off:  (2, 160, 160) - Sub-cell floating point offset
        mask: (1, 160, 160) - 1.0 at person centers, 0.0 elsewhere (for masked loss)
    """
    hm = np.zeros((1, out, out), dtype=np.float32)
    wh = np.zeros((2, out, out), dtype=np.float32)
    off = np.zeros((2, out, out), dtype=np.float32)
    mask = np.zeros((1, out, out), dtype=np.float32)

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box[:4]
        w_cell = (x2 - x1) / stride
        h_cell = (y2 - y1) / stride

        if w_cell <= 0 or h_cell <= 0:
            continue

        cx_cell = (x1 + x2) / 2.0 / stride
        cy_cell = (y1 + y2) / 2.0 / stride

        ix = int(np.clip(int(cx_cell), 0, out - 1))
        iy = int(np.clip(int(cy_cell), 0, out - 1))

        # Radius floored at 1 so tiny 15px people still have a distinct 3x3 Gaussian bump
        r = max(1, int(gaussian_radius(h_cell, w_cell)))
        draw_gaussian(hm[0], ix, iy, r)

        hm[0, iy, ix] = 1.0  # Center is exactly 1.0
        wh[:, iy, ix] = (w_cell, h_cell)  # Box dimensions
        off[:, iy, ix] = (cx_cell - ix, cy_cell - iy)  # Sub-cell offset
        mask[0, iy, ix] = 1.0  # Positive sample mask

    return hm, wh, off, mask

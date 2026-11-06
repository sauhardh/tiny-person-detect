"""
src/infer/decode.py — CenterNet Heatmap & Box Decoder
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def decode(
    hm_logits: torch.Tensor,
    wh: torch.Tensor,
    off: torch.Tensor,
    stride: int = 4,
    K: int = 100,
    thr: float = 0.05,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Decodes CenterNet heatmap predictions into bounding boxes.

    1. Applies sigmoid to obtain probabilities.
    2. Performs 3x3 max pooling NMS to identify local heatmap peaks.
    3. Selects top-K highest confidence peaks.
    4. Computes pixel bounding boxes:
       cx = (x_cell + off_x) * stride
       cy = (y_cell + off_y) * stride
       w  = wh_x * stride
       h  = wh_y * stride
       [x1, y1, x2, y2] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    Args:
        hm_logits: (B, 1, H, W) heatmap logits before sigmoid.
        wh:        (B, 2, H, W) box size in stride-4 cells.
        off:       (B, 2, H, W) sub-cell floating offset.
        stride:    detector downsampling factor (default: 4).
        K:         maximum detections to return per image (default: 100).
        thr:       minimum confidence threshold to retain detection.

    Returns:
        List of tuples per batch element: (boxes_xyxy, scores)
        - boxes_xyxy: (N, 4) float32 tensor of [x1, y1, x2, y2] in tile coordinates
        - scores:     (N,) float32 tensor of confidence scores
    """
    hm = torch.sigmoid(hm_logits.float())

    # 3x3 Max-pooling NMS: keep only local maxima
    hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hmax == hm).float()
    hm = hm * keep

    B, _, H, W = hm.shape
    # Flatten spatial dims to find top-K peaks
    scores, idx = hm.view(B, -1).topk(min(K, H * W))
    ys = (idx // W).float()
    xs = (idx % W).float()

    results = []
    for b in range(B):
        mask = scores[b] > thr
        ix = idx[b][mask]

        if ix.numel() == 0:
            results.append((
                torch.zeros((0, 4), dtype=torch.float32, device=hm.device),
                torch.zeros((0,), dtype=torch.float32, device=hm.device),
            ))
            continue

        w = wh[b, 0].reshape(-1)[ix]
        h = wh[b, 1].reshape(-1)[ix]
        ox = off[b, 0].reshape(-1)[ix]
        oy = off[b, 1].reshape(-1)[ix]

        cx = (xs[b][mask] + ox) * stride
        cy = (ys[b][mask] + oy) * stride
        bw = w * stride
        bh = h * stride

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        results.append((boxes, scores[b][mask]))

    return results

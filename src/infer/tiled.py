"""
src/infer/tiled.py — Sliced Overlapping Tiled Inference for Full-Resolution Drone Frames
"""

from typing import List, Tuple
import numpy as np
import torch
import torchvision

from src.infer.decode import decode

# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tile_origins(L: int, T: int, stride: int) -> List[int]:
    """
    Computes 1D grid origins covering length L with tiles of size T and step stride.
    Guarantees the last tile reaches exactly L - T to cover frame edges.
    """
    if L <= T:
        return [0]
    coords = list(range(0, L - T + 1, stride))
    if coords[-1] + T < L:
        coords.append(L - T)
    return coords


def pad_to(crop: np.ndarray, T: int) -> np.ndarray:
    """Zero-pads a crop if smaller than T x T."""
    h, w = crop.shape[:2]
    if h == T and w == T:
        return crop
    padded = np.zeros((T, T, 3), dtype=np.uint8)
    padded[:h, :w] = crop
    return padded


def to_batch(crops: List[np.ndarray]) -> torch.Tensor:
    """Converts a list of HxWx3 uint8 crops into a normalized (B, 3, H, W) float tensor."""
    arr = (np.stack(crops).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    img: np.ndarray,
    T: int = 640,
    overlap: float = 0.25,
    thr: float = 0.20,
    nms_iou: float = 0.5,
    border_margin: int = 16,
    batch_size: int = 16,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs full-frame inference using overlapping sliding tiles and cross-tile NMS.

    Args:
        model:         Trained Detector instance.
        img:           (H, W, 3) full-frame RGB numpy array (e.g. 5472 x 3078).
        T:             Tile size (default: 640).
        overlap:       Tile overlap percentage (default: 0.25, step = 480 px).
        thr:           Minimum confidence threshold for tile detections (default: 0.20).
        nms_iou:       IoU threshold for global cross-tile Non-Maximum Suppression (default: 0.5).
        border_margin: Suppress detections within this margin of internal tile edges (default: 16).
        batch_size:    Mini-batch size of tiles processed simultaneously.
        device:        PyTorch device to run model on.

    Returns:
        boxes:  (N, 4) float32 numpy array of [x1, y1, x2, y2] in full-frame coordinates.
        scores: (N,) float32 numpy array of detection confidence scores.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    H, W = img.shape[:2]
    stride = int(T * (1.0 - overlap))

    # Generate 2D tile grid origins (tx, ty)
    x_coords = tile_origins(W, T, stride)
    y_coords = tile_origins(H, T, stride)
    coords = [(x, y) for y in y_coords for x in x_coords]

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []

    # Batch process tiles through the detector
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        crops = [pad_to(img[y : y + T, x : x + T], T) for x, y in chunk]
        x_tensor = to_batch(crops).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(x_tensor)

        dets = decode(
            out["hm"].float(),
            out["wh"].float(),
            out["off"].float(),
            stride=4,
            thr=thr,
        )

        # Shift tile detections into full-frame coordinates with border margin suppression
        for (boxes, scores), (tx, ty) in zip(dets, chunk):
            if len(boxes) > 0:
                if border_margin > 0:
                    cxs = (boxes[:, 0] + boxes[:, 2]) / 2.0
                    cys = (boxes[:, 1] + boxes[:, 3]) / 2.0
                    valid = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
                    if tx > 0:
                        valid &= (cxs >= border_margin)
                    if tx + T < W:
                        valid &= (cxs <= T - border_margin)
                    if ty > 0:
                        valid &= (cys >= border_margin)
                    if ty + T < H:
                        valid &= (cys <= T - border_margin)
                    boxes = boxes[valid]
                    scores = scores[valid]

                if len(boxes) > 0:
                    shifted = boxes.clone()
                    shifted[:, [0, 2]] += tx
                    shifted[:, [1, 3]] += ty
                    all_boxes.append(shifted)
                    all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    cat_boxes = torch.cat(all_boxes, dim=0)
    cat_scores = torch.cat(all_scores, dim=0)

    # Cross-tile NMS to merge duplicate detections across tile borders
    keep_indices = torchvision.ops.nms(cat_boxes, cat_scores, nms_iou)

    final_boxes = cat_boxes[keep_indices].cpu().numpy()
    final_scores = cat_scores[keep_indices].cpu().numpy()

    return final_boxes, final_scores

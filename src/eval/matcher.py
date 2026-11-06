"""
src/eval/matcher.py — Bounding Box Matching & IoU Computation
"""

import numpy as np


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of bounding boxes.

    Args:
        a: (N, 4) numpy array of [x1, y1, x2, y2].
        b: (M, 4) numpy array of [x1, y1, x2, y2].

    Returns:
        (N, M) matrix where element (i, j) is IoU between box a[i] and b[j].
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    area_a = np.maximum(0.0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
    area_b = np.maximum(0.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
):
    """
    Greedy matching of detections to ground truth boxes ordered by confidence score.

    Args:
        pred_boxes:  (N, 4) predicted bounding boxes.
        pred_scores: (N,) confidence scores.
        gt_boxes:    (M, 4) ground truth boxes.
        iou_thr:     minimum IoU for a match (default: 0.5).

    Returns:
        sorted_scores: (N,) predictions sorted descending by score.
        tp_flags:      (N,) bool array indicating True Positive (True) or False Positive (False).
        gt_matched:    (M,) highest score matching each ground truth box (-1.0 if missed).
    """
    if len(pred_scores) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.full(len(gt_boxes), -1.0, dtype=np.float32),
        )

    order = np.argsort(-pred_scores)
    pb = pred_boxes[order]
    ps = pred_scores[order]

    tp = np.zeros(len(ps), dtype=bool)
    gt_score = np.full(len(gt_boxes), -1.0, dtype=np.float32)

    if len(ps) > 0 and len(gt_boxes) > 0:
        ious = box_iou_np(pb, gt_boxes)
        gt_taken = np.zeros(len(gt_boxes), dtype=bool)

        for i in range(len(ps)):
            cand_ious = ious[i].copy()
            cand_ious[gt_taken] = -1.0
            best_gt_idx = int(cand_ious.argmax())

            if cand_ious[best_gt_idx] >= iou_thr:
                tp[i] = True
                gt_taken[best_gt_idx] = True
                gt_score[best_gt_idx] = ps[i]

    return ps, tp, gt_score

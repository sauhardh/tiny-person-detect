"""
src/eval/metrics.py — Precision, Recall, mAP50 & Distance-Stratified Evaluation
"""

from typing import Any, Dict
import numpy as np

from src.eval.matcher import match_image


def average_precision(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    """
    Computes AP50 using standard all-point interpolated precision-recall curve.

    Args:
        scores: (N,) confidence scores.
        tps:    (N,) bool array indicating True Positives.
        n_gt:   Total number of ground truth instances across the dataset.

    Returns:
        float AP50 in range [0.0, 1.0].
    """
    if len(scores) == 0 or n_gt == 0:
        return 0.0

    order = np.argsort(-scores)
    tp = tps[order].astype(np.float64)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # Pad edges: recall [0, ..., 1], precision [1, ..., 0]
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])

    # Monotonic decreasing envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))
    return max(0.0, min(1.0, ap))


def evaluate(
    predictions: Dict[str, Any],
    ground_truths: Dict[str, Any],
    conf_thr: float = 0.25,
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation matching predictions against ground truth labels.

    Args:
        predictions: Dict mapping image_id/path to (boxes: Nx4, scores: N)
        ground_truths: Dict mapping image_id/path to {"boxes": Mx4, "dist": list of flight distances}
        conf_thr: Detection threshold for reporting precision, recall, and FP/image.
        iou_thr: Minimum IoU threshold for a True Positive match (default: 0.5).

    Returns:
        dict containing:
          - precision: TP / (TP + FP) at conf_thr
          - recall: TP / total_gt at conf_thr
          - fp_per_image: FP count / total test images
          - ap50: Average Precision @ IoU 0.50
          - recall_by_dist: Breakdown of recall at 10m, 30m, 50m, 70m, 90m
    """
    all_scores = []
    all_tps = []
    total_gt = 0

    tp_at_thr = 0
    fp_at_thr = 0
    # distance -> [found_count, total_count]
    dist_stats: Dict[int, list] = {}

    for img_key, gt_data in ground_truths.items():
        gt_boxes = np.asarray(gt_data["boxes"], dtype=np.float32)
        gt_dists = gt_data.get("dist", [0] * len(gt_boxes))
        total_gt += len(gt_boxes)

        pred_boxes, pred_scores = predictions.get(
            img_key,
            (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)),
        )
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
        pred_scores = np.asarray(pred_scores, dtype=np.float32)

        s_sorted, tp_flags, gt_matched = match_image(
            pred_boxes,
            pred_scores,
            gt_boxes,
            iou_thr=iou_thr,
        )

        all_scores.append(s_sorted)
        all_tps.append(tp_flags)

        # Confidence threshold metrics
        above_thr = s_sorted >= conf_thr
        tp_at_thr += int(tp_flags[above_thr].sum())
        fp_at_thr += int((~tp_flags[above_thr]).sum())

        for dist_val, match_score in zip(gt_dists, gt_matched):
            d_key = int(dist_val)
            if d_key not in dist_stats:
                dist_stats[d_key] = [0, 0]
            dist_stats[d_key][1] += 1
            if match_score >= conf_thr:
                dist_stats[d_key][0] += 1

    cat_scores = (
        np.concatenate(all_scores) if all_scores else np.zeros((0,), dtype=np.float32)
    )
    cat_tps = np.concatenate(all_tps) if all_tps else np.zeros((0,), dtype=bool)

    precision = float(tp_at_thr / max(tp_at_thr + fp_at_thr, 1))
    recall = float(tp_at_thr / max(total_gt, 1))
    fp_per_image = float(fp_at_thr / max(len(ground_truths), 1))
    ap50 = average_precision(cat_scores, cat_tps, total_gt)

    recall_by_dist = {}
    for d, (f, t) in sorted(dist_stats.items()):
        recall_by_dist[d] = {
            "found": f,
            "total": t,
            "recall": float(f / max(t, 1)),
        }

    return {
        "precision": precision,
        "recall": recall,
        "fp_per_image": fp_per_image,
        "ap50": ap50,
        "total_gt": total_gt,
        "tp": tp_at_thr,
        "fp": fp_at_thr,
        "recall_by_dist": recall_by_dist,
    }

"""
src/train/losses.py — CenterNet Loss Formulation
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    Penalty-reduced pixelwise focal loss from CenterNet (Zhou et al.).

    Args:
        logits: (B, 1, H, W) raw unnormalized model predictions (before sigmoid).
        gt:     (B, 1, H, W) ground truth Gaussian heatmap in [0, 1].
        alpha:  Focusing parameter (default: 2.0).
        beta:   Penalty-reduction parameter for near-center pixels (default: 4.0).

    Returns:
        Scalar tensor containing focal loss averaged over positive samples.
    """
    # Always compute loss in float32 (even under AMP fp16) to avoid overflow
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1.0 - 1e-4)

    # Positive mask: exact center peaks (gt == 1.0)
    pos = gt.eq(1.0).float()
    # Negative mask: surrounding background (gt < 1.0)
    neg = 1.0 - pos

    # Positives: standard focal loss focusing on hard detections
    pos_loss = torch.log(pred) * ((1.0 - pred) ** alpha) * pos

    # Negatives: down-weight near-center pixels with (1 - gt)^beta
    neg_loss = torch.log(1.0 - pred) * (pred**alpha) * ((1.0 - gt) ** beta) * neg

    num_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss computed ONLY at positive center cells (where mask == 1.0).

    Args:
        pred:   (B, 2, H, W) predicted sizes (wh) or sub-pixel offsets (off).
        target: (B, 2, H, W) ground truth sizes or offsets.
        mask:   (B, 1, H, W) 1.0 where a person center exists, 0.0 elsewhere.

    Returns:
        Scalar tensor containing mean L1 loss on positive cells.
    """
    # Expand mask from (B, 1, H, W) to (B, 2, H, W) to match (w, h) or (dx, dy)
    m = mask.expand_as(pred).float()
    loss = torch.abs(pred.float() - target.float()) * m
    num_pos = m.sum().clamp(min=1e-4)
    return loss.sum() / num_pos


def compute_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    w_wh: float = 0.1,
    w_off: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combines Heatmap Focal Loss, Size L1, and Offset L1:
        Total Loss = L_hm + 0.1 * L_wh + 1.0 * L_off

    Args:
        out:   Output dictionary from Detector containing "hm", "wh", "off".
        batch: Ground truth dictionary from TileDataset containing "hm", "wh", "off", "mask".
        w_wh:  Loss weight for size head (CenterNet paper default: 0.1).
        w_off: Loss weight for offset head (CenterNet paper default: 1.0).

    Returns:
        total_loss: Scalar PyTorch tensor for backprop.
        loss_dict:  Dictionary of individual loss values for logging.
    """
    l_hm = focal_loss(out["hm"], batch["hm"])
    l_wh = masked_l1(out["wh"], batch["wh"], batch["mask"])
    l_off = masked_l1(out["off"], batch["off"], batch["mask"])

    total_loss = l_hm + (w_wh * l_wh) + (w_off * l_off)

    loss_dict = {
        "loss": total_loss.item(),
        "hm": l_hm.item(),
        "wh": l_wh.item(),
        "off": l_off.item(),
    }
    return total_loss, loss_dict

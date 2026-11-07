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


def ciou_loss(
    pred_wh: torch.Tensor,
    pred_off: torch.Tensor,
    target_wh: torch.Tensor,
    target_off: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Complete-IoU (CIoU) loss for CenterNet box predictions.
    Directly optimizes bounding box overlap, center distance, and aspect ratio.
    """
    pos = mask.squeeze(1) > 0.5  # (B, H, W)
    if not pos.any():
        return torch.tensor(0.0, device=pred_wh.device, dtype=pred_wh.dtype, requires_grad=True)

    B, _, H, W = pred_wh.shape
    device = pred_wh.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=pred_wh.dtype),
        torch.arange(W, device=device, dtype=pred_wh.dtype),
        indexing="ij",
    )
    xs = xs.unsqueeze(0).expand(B, -1, -1)[pos]
    ys = ys.unsqueeze(0).expand(B, -1, -1)[pos]

    p_w = F.relu(pred_wh[:, 0][pos].float()) + 1e-4
    p_h = F.relu(pred_wh[:, 1][pos].float()) + 1e-4
    p_ox = pred_off[:, 0][pos].float()
    p_oy = pred_off[:, 1][pos].float()

    t_w = target_wh[:, 0][pos].float()
    t_h = target_wh[:, 1][pos].float()
    t_ox = target_off[:, 0][pos].float()
    t_oy = target_off[:, 1][pos].float()

    p_cx = xs + p_ox
    p_cy = ys + p_oy
    t_cx = xs + t_ox
    t_cy = ys + t_oy

    p_x1 = p_cx - p_w / 2.0
    p_y1 = p_cy - p_h / 2.0
    p_x2 = p_cx + p_w / 2.0
    p_y2 = p_cy + p_h / 2.0

    t_x1 = t_cx - t_w / 2.0
    t_y1 = t_cy - t_h / 2.0
    t_x2 = t_cx + t_w / 2.0
    t_y2 = t_cy + t_h / 2.0

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)

    union = (p_w * p_h) + (t_w * t_h) - inter + 1e-7
    iou = inter / union

    rho2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    v = (4.0 / (torch.pi ** 2)) * (torch.atan(t_w / (t_h + 1e-7)) - torch.atan(p_w / (p_h + 1e-7))) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-7)

    ciou = iou - (rho2 / c2) - (alpha * v)
    loss = 1.0 - ciou
    return loss.mean()


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
    w_ciou: float = 1.0,
    w_wh: float = 0.1,
    w_off: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combines Heatmap Focal Loss, CIoU Box Loss, Size L1, and Offset L1:
        Total Loss = L_hm + 1.0 * L_ciou + 0.1 * L_wh + 1.0 * L_off

    Args:
        out:    Output dictionary from Detector containing "hm", "wh", "off".
        batch:  Ground truth dictionary from TileDataset containing "hm", "wh", "off", "mask".
        w_ciou: Loss weight for CIoU box loss (default: 1.0).
        w_wh:   Loss weight for auxiliary size L1 head (default: 0.1).
        w_off:  Loss weight for offset head (default: 1.0).

    Returns:
        total_loss: Scalar PyTorch tensor for backprop.
        loss_dict:  Dictionary of individual loss values for logging.
    """
    l_hm = focal_loss(out["hm"], batch["hm"])
    l_ciou = ciou_loss(out["wh"], out["off"], batch["wh"], batch["off"], batch["mask"])
    l_wh = masked_l1(out["wh"], batch["wh"], batch["mask"])
    l_off = masked_l1(out["off"], batch["off"], batch["mask"])

    total_loss = l_hm + (w_ciou * l_ciou) + (w_wh * l_wh) + (w_off * l_off)

    loss_dict = {
        "loss": total_loss.item(),
        "hm": l_hm.item(),
        "ciou": l_ciou.item(),
        "wh": l_wh.item(),
        "off": l_off.item(),
    }
    return total_loss, loss_dict

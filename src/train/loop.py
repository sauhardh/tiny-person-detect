"""
src/train/loop.py — Training Engine with AMP, Cosine LR, Checkpointing & Overfit Sanity Test
"""

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from src.detector import Detector
from src.pre.data import TileDataset
from src.train.losses import compute_loss


def set_seed(seed: int = 42):
    """Ensures repeatable training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_dev(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Moves batch dictionary tensors to target device."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluates validation loss across the val split."""
    model.eval()
    tot_loss, tot_hm, tot_wh, tot_off = 0.0, 0.0, 0.0, 0.0
    num_batches = len(val_loader)

    for batch in val_loader:
        batch = to_dev(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(batch["image"])
            loss, parts = compute_loss(out, batch)

        tot_loss += loss.item()
        tot_hm += parts["hm"]
        tot_wh += parts["wh"]
        tot_off += parts["off"]

    n = max(num_batches, 1)
    metrics = {
        "val_loss": tot_loss / n,
        "val_hm": tot_hm / n,
        "val_wh": tot_wh / n,
        "val_off": tot_off / n,
    }
    return metrics["val_loss"], metrics


def overfit_test(
    manifest_csv: str = "data/meta/all_tiles.csv",
    root: str = None,
    num_tiles: int = 16,
    steps: int = 250,
    lr: float = 1e-3,
):
    """
    Karpathy Sanity Test:
    Takes 16 tiles and verifies that the model can drive training loss toward 0.
    If loss does not drop significantly (< 10% of initial), there is a bug in
    data target generation, loss math, or network heads.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"\n[Overfit Test] Running on {device} for {steps} steps with {num_tiles} tiles..."
    )

    # Load 16 fixed tiles without data augmentation
    ds = TileDataset(manifest_csv, root=root, split="train", train=False)
    # Pick tiles with people so there is actual loss to optimize
    person_indices = []
    for i, (_, lbl_path) in enumerate(ds.samples):
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            person_indices.append(i)
        if len(person_indices) >= num_tiles:
            break

    sub_ds = Subset(ds, person_indices)
    loader = DataLoader(sub_ds, batch_size=num_tiles, shuffle=False)
    batch = next(iter(loader))
    batch = to_dev(batch, device)

    model = Detector("resnet18", pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    initial_loss = None
    for step in range(1, steps + 1):
        model.train()
        out = model(batch["image"])
        loss, parts = compute_loss(out, batch)

        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == 1:
            print(
                f"  Step {step:>3d}/{steps} | Total: {loss.item():.4f} "
                f"(HM: {parts['hm']:.4f}, WH: {parts['wh']:.4f}, OFF: {parts['off']:.4f})"
            )

    drop_ratio = (loss.item() / initial_loss) * 100.0
    print(
        f"\nInitial loss: {initial_loss:.4f} -> Final loss: {loss.item():.4f} ({drop_ratio:.1f}% of starting)"
    )
    assert loss.item() < initial_loss * 0.25, (
        "Overfit test failed! Model could not memorize 16 tiles."
    )
    print(
        "-> OVERFIT TEST PASSED! The data, model, and loss formulation are bug-free.\n"
    )


def train(cfg: Dict[str, Any]):
    """
    Main Training Loop with AMP, Warmup + Cosine Decay, and Safe Resuming.
    """
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.get("out", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets & Loaders
    manifest = cfg.get("manifest", "data/meta/all_tiles.csv")
    root = cfg.get("root")
    train_ds = TileDataset(
        manifest, root=root, split="train", source=cfg.get("source"), train=True
    )
    val_ds = TileDataset(
        manifest, root=root, split="val", source=cfg.get("source"), train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("bs", 32),
        shuffle=True,
        num_workers=cfg.get("workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("bs", 32),
        shuffle=False,
        num_workers=cfg.get("workers", 4),
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Model & Optimizer
    model = Detector(
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=cfg.get("pretrained", True),
        pretrained_path=cfg.get("pretrained_path"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 2e-4),
        weight_decay=cfg.get("wd", 1e-4),
    )

    # 3. Warmup + Cosine Annealing Scheduler (stepped per iteration)
    epochs = cfg.get("epochs", 60)
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)  # 1 full epoch warmup

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # 4. Resume Checkpointing
    last_pt = out_dir / "last.pt"
    best_pt = out_dir / "best.pt"
    start_epoch, best_val_loss = 0, float("inf")

    if last_pt.exists():
        state = torch.load(last_pt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state.get("best", float("inf"))
        print(
            f"Resumed successfully from epoch {start_epoch} (best val loss: {best_val_loss:.4f})"
        )

    # 5. Training Loop
    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            batch = to_dev(batch, device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                out = model(batch["image"])
                loss, parts = compute_loss(out, batch)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_parts = validate(model, val_loader, device)

        print(
            f"Epoch [{epoch + 1:>2d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} (HM: {val_parts['val_hm']:.4f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_pt)
            print(f"  -> Best model saved to {best_pt}")

        # Save Last Checkpoint (for resume)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best": best_val_loss,
            },
            last_pt,
        )

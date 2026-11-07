"""
kaggle_train.py — Standalone Kaggle Training & Evaluation Script

Can be pasted directly into a Kaggle Notebook code cell or run via command line:
    python notebooks/kaggle_train.py
"""

import os
import sys
from pathlib import Path

# Detect Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    print("Detected Kaggle Environment!")
    # In Kaggle:
    # Code is either cloned to /kaggle/working/repo or current dir
    # Input datasets live in /kaggle/input/<dataset-name>/...
    # Output checkpoints MUST be written to /kaggle/working/
    ROOT_DIR = Path("/kaggle/working")
    REPO_DIR = Path("/kaggle/working/tiny-person-detector")
    if REPO_DIR.exists():
        sys.path.insert(0, str(REPO_DIR))
        sys.path.insert(0, str(REPO_DIR / "src"))
    else:
        sys.path.insert(0, ".")
        sys.path.insert(0, "src")

    # Locate dataset path in /kaggle/input
    # Search for all_tiles.csv in /kaggle/input
    import glob
    candidates = glob.glob("/kaggle/input/**/all_tiles.csv", recursive=True)
    if candidates:
        MANIFEST_CSV = candidates[0]
        # Data root is the directory containing the "data" folder or parent
        DATA_ROOT = str(Path(MANIFEST_CSV).parent.parent.parent)
    else:
        # Default fallback
        MANIFEST_CSV = "/kaggle/input/nomad-tiles/data/meta/all_tiles.csv"
        DATA_ROOT = "/kaggle/input/nomad-tiles"

    OUT_DIR = "/kaggle/working/checkpoints"
else:
    print("Detected Local Environment!")
    sys.path.insert(0, ".")
    sys.path.insert(0, "src")
    MANIFEST_CSV = "data/meta/all_tiles.csv"
    DATA_ROOT = "."
    OUT_DIR = "checkpoints"

from src.train.loop import train, overfit_test


def main():
    print(f"Manifest CSV : {MANIFEST_CSV}")
    print(f"Data Root    : {DATA_ROOT}")
    print(f"Output Dir   : {OUT_DIR}")

    # Step 1: Optional sanity check on Kaggle (verifies loss converges on GPU)
    # overfit_test(manifest_csv=MANIFEST_CSV, root=DATA_ROOT, num_tiles=16, steps=100)

    # Step 2: Full Training Run (60 Epochs with AMP & Cosine Annealing)
    config = {
        "manifest": MANIFEST_CSV,
        "root": DATA_ROOT,
        "out": OUT_DIR,
        "backbone": "resnet18",
        "pretrained": True,
        "source": None,          # None = Mixed (NOMAD + SAR), or 'nomad', or 'sar'
        "epochs": 60,
        "patience": 10,          # Early stopping to prevent overfitting
        "bs": 32,                # Batch size 32 fits on 16GB T4 GPU
        "lr": 2e-4,
        "wd": 1e-4,
        "workers": 4,
        "seed": 42,
    }

    train(config)


if __name__ == "__main__":
    main()

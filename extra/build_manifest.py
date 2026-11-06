import os
import csv
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "data"
OUT_CSV = ROOT / "meta" / "all_tiles.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

rows = []
splits = ["train", "val", "test"]

# 1. Index NOMAD Tiles
for split in splits:
    img_files = sorted(
        glob.glob(str(ROOT / "nomad_tiles" / "images" / split / "*.jpg"))
    )
    for img_p in img_files:
        lbl_p = img_p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        is_bg = "_bg" in os.path.basename(img_p)
        rows.append(
            {
                "image_path": os.path.relpath(img_p, SCRIPT_DIR),
                "label_path": os.path.relpath(lbl_p, SCRIPT_DIR),
                "split": split,
                "source": "nomad",
                "is_bg": 1 if is_bg else 0,
            }
        )

# 2. Index Search-and-Rescue (SAR)
for split in splits:
    img_files = sorted(
        glob.glob(str(ROOT / "search-and-rescue" / split / "images" / "*.jpg"))
    )
    for img_p in img_files:
        stem = Path(img_p).stem
        lbl_p = str(ROOT / "search-and-rescue" / split / "labels" / f"{stem}.txt")
        is_bg = os.path.exists(lbl_p) and os.path.getsize(lbl_p) == 0
        rows.append(
            {
                "image_path": os.path.relpath(img_p, SCRIPT_DIR),
                "label_path": os.path.relpath(lbl_p, SCRIPT_DIR),
                "split": split,
                "source": "sar",
                "is_bg": 1 if is_bg else 0,
            }
        )

# Write Master CSV
fieldnames = ["image_path", "label_path", "split", "source", "is_bg"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Master manifest written: {len(rows)} total samples in {OUT_CSV}")

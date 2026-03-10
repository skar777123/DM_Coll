"""
ML_Model/yolo/prepare_dataset.py
─────────────────────────────────
Dataset preparation script for custom YOLOv8 training.

What it does
────────────
1. Downloads a COCO subset (only the 6 relevant classes) using fiftyone
   — or falls back to a manual download path if fiftyone is unavailable.
2. Converts COCO JSON annotations → YOLO TXT format.
3. Applies class-ID remapping to our 6-class schema.
4. Generates train / val / test splits.
5. Prints a dataset summary.

Classes mapped (COCO id → our id)
──────────────────────────────────
  COCO  2 (car)        → 0
  COCO  7 (truck)      → 1
  COCO  3 (motorcycle) → 2
  COCO  5 (bus)        → 3
  COCO  0 (person)     → 4
  COCO  1 (bicycle)    → 5

Usage
─────
  pip install fiftyone
  python ML_Model/yolo/prepare_dataset.py [--max-samples 5000]

  Or, if you have your own images + COCO annotations:
  python ML_Model/yolo/prepare_dataset.py --coco-json /path/to/annotations.json \\
         --images-dir /path/to/images --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ── COCO class ID → our class ID ────────────────────────────────────────────
COCO_CLASS_MAP = {
    2: 0,   # car
    7: 1,   # truck
    3: 2,   # motorcycle
    5: 3,   # bus
    0: 4,   # person
    1: 5,   # bicycle
}
CLASS_NAMES = ["car", "truck", "motorcycle", "bus", "person", "bicycle"]

DATASET_ROOT = Path("ML_Model/yolo/dataset")


# ─────────────────────────────────────────────────────────────────────────────
#  COCO JSON → YOLO TXT
# ─────────────────────────────────────────────────────────────────────────────

def coco_to_yolo(
    coco_json_path: str,
    images_src_dir: str,
    split: str,
    max_samples: int | None = None,
) -> int:
    """
    Convert COCO annotation JSON + images folder → YOLO dataset split.
    Returns number of images converted.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    img_dir   = DATASET_ROOT / "images" / split
    lbl_dir   = DATASET_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Build image_id → image_info map
    id_to_info = {img["id"]: img for img in coco["images"]}

    # Build image_id → list of annotations
    id_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        coco_cat = ann.get("category_id", -1) - 1   # COCO is 1-indexed
        if coco_cat not in COCO_CLASS_MAP:
            continue
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_ids = list(id_to_anns.keys())
    random.shuffle(img_ids)
    if max_samples:
        img_ids = img_ids[:max_samples]

    converted = 0
    for img_id in img_ids:
        info  = id_to_info.get(img_id)
        if info is None:
            continue

        src_img = Path(images_src_dir) / info["file_name"]
        dst_img = img_dir / info["file_name"]
        if not src_img.exists():
            continue

        shutil.copy2(src_img, dst_img)

        W, H = info["width"], info["height"]
        label_lines = []
        for ann in id_to_anns[img_id]:
            coco_cat = ann["category_id"] - 1
            our_cls  = COCO_CLASS_MAP[coco_cat]
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            nw = bw / W
            nh = bh / H
            label_lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = Path(info["file_name"]).stem
        (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines))
        converted += 1

    return converted


# ─────────────────────────────────────────────────────────────────────────────
#  fiftyone downloader  (automatic)
# ─────────────────────────────────────────────────────────────────────────────

def download_via_fiftyone(max_samples: int = 3000):
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("fiftyone not installed. Run: pip install fiftyone")
        print("Or provide --coco-json and --images-dir manually.")
        return False

    raw_dir = DATASET_ROOT / "_raw_coco"
    raw_dir.mkdir(parents=True, exist_ok=True)

    classes = CLASS_NAMES
    per_split = {"train": int(max_samples * 0.75), "val": int(max_samples * 0.2), "test": int(max_samples * 0.05)}

    for split, n in per_split.items():
        print(f"  Downloading {n} COCO samples for [{split}] …")
        try:
            ds = foz.load_zoo_dataset(
                "coco-2017", split=split,
                label_types=["detections"],
                classes=classes,
                max_samples=n,
                dataset_dir=str(raw_dir / split),
            )
            # Export in COCO format for our converter
            export_dir = raw_dir / split / "export"
            ds.export(
                export_dir=str(export_dir),
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
            )
            n_conv = coco_to_yolo(
                coco_json_path=str(export_dir / "labels.json"),
                images_src_dir=str(export_dir / "data"),
                split=split,
                max_samples=n,
            )
            print(f"  ✔ {split}: {n_conv} images converted")
        except Exception as exc:
            print(f"  ⚠ Failed for split [{split}]: {exc}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n── Dataset Summary ─────────────────────────────────")
    for split in ("train", "val", "test"):
        img_dir = DATASET_ROOT / "images" / split
        lbl_dir = DATASET_ROOT / "labels" / split
        n_img = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split:6s}  images: {n_img:5d}   labels: {n_lbl:5d}")
    print(f"  Root: {DATASET_ROOT.resolve()}")
    print("────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=4000,
                        help="Total images to download (split 75/20/5)")
    parser.add_argument("--coco-json",   type=str, default=None,
                        help="Path to existing COCO annotations JSON (manual mode)")
    parser.add_argument("--images-dir",  type=str, default=None,
                        help="Path to existing COCO images dir (manual mode)")
    parser.add_argument("--split",       type=str, default="train",
                        choices=["train","val","test"])
    args = parser.parse_args()

    if args.coco_json and args.images_dir:
        # Manual mode
        print(f"Manual COCO → YOLO conversion  [{args.split}]")
        n = coco_to_yolo(args.coco_json, args.images_dir, args.split, args.max_samples)
        print(f"Converted {n} images.")
    else:
        # Auto download via fiftyone
        print(f"Auto-downloading COCO subset via fiftyone ({args.max_samples} samples)…")
        ok = download_via_fiftyone(args.max_samples)
        if not ok:
            sys.exit(1)

    print_summary()
    print("\nDataset ready. Next step:")
    print("  python ML_Model/yolo/train_yolo.py")


if __name__ == "__main__":
    main()

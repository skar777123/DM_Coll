"""
ML_Model/yolo/train_yolo.py
────────────────────────────
Custom YOLOv8 Fine-Tuning Script for BlindSpotGuard.

Starts from a pretrained YOLOv8n checkpoint and fine-tunes on our
6-class blind-spot vehicle dataset (car, truck, motorcycle, bus, person, bicycle).

Features
────────
  • Pi-optimised model size  (YOLOv8n — nano, ~3.2M params)
  • Custom augmentation profile tuned for dashcam blind-spot imagery
  • Mixed precision training (FP16 on GPU, FP32 on CPU/Pi)
  • Mosaic + MixUp + HSV augmentation
  • Auto-export to ONNX + TFLite (INT8) for Pi deployment
  • Confusion matrix + PR curve saved automatically

Usage
─────
  # Standard fine-tune on GPU
  python ML_Model/yolo/train_yolo.py

  # Resume from checkpoint
  python ML_Model/yolo/train_yolo.py --resume

  # Quick test (5 epochs)
  python ML_Model/yolo/train_yolo.py --epochs 5 --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ML_Model.yolo.augment import AUGMENT_ARGS

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────

YOLO_DIR    = Path("ML_Model/yolo")
DATASET_CFG = str(YOLO_DIR / "dataset.yaml")
RUNS_DIR    = str(YOLO_DIR / "runs")
EXPORT_DIR  = str(Path("ML_Model/saved_models"))

# ─────────────────────────────────────────────────────────────────────────────
#  Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_ARGS = {
    # Data
    "data":         DATASET_CFG,
    "imgsz":        320,           # 320×320 — faster on Pi, still accurate
    "batch":        16,            # adjust if GPU OOM
    "workers":      4,

    # Optimisation
    "epochs":       100,
    "optimizer":    "AdamW",
    "lr0":          0.001,
    "lrf":          0.01,          # final LR = lr0 * lrf
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_bias_lr": 0.1,

    # Loss weights (slightly emphasise objectness for small objects)
    "box":          7.5,
    "cls":          0.5,
    "dfl":          1.5,

    # Output
    "project":      RUNS_DIR,
    "name":         "blindspot_yolo",
    "exist_ok":     True,
    "save_period":  10,            # checkpoint every 10 epochs
    "plots":        True,
    "val":          True,

    # Hardware
    "device":       "0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "amp":          True,          # automatic mixed precision
    "cache":        True,          # cache images for speed

    # Augmentation
    **AUGMENT_ARGS,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Export helper
# ─────────────────────────────────────────────────────────────────────────────

def export_models(run_dir: Path) -> None:
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print("  ⚠ best.pt not found — cannot export.")
        return

    os.makedirs(EXPORT_DIR, exist_ok=True)

    print("\n── Exporting models ───────────────────────────────────")

    model = YOLO(str(best_pt))

    # 1. ONNX (GPU accelerated inference, works on Pi via ONNX Runtime)
    onnx_path = model.export(format="onnx", imgsz=320, simplify=True, dynamic=False)
    dst_onnx  = Path(EXPORT_DIR) / "blindspot_yolo.onnx"
    shutil.copy2(onnx_path, dst_onnx)
    print(f"  ONNX   → {dst_onnx}  ({dst_onnx.stat().st_size / 1e6:.1f} MB)")

    # 2. TFLite INT8 (best for Raspberry Pi CPU)
    tflite_path = model.export(format="tflite", imgsz=320, int8=True)
    if tflite_path:
        dst_tflite = Path(EXPORT_DIR) / "blindspot_yolo_int8.tflite"
        shutil.copy2(tflite_path, dst_tflite)
        print(f"  TFLite → {dst_tflite}  ({dst_tflite.stat().st_size / 1e6:.1f} MB)")

    # Also save the raw weights
    dst_pt = Path(EXPORT_DIR) / "blindspot_yolo.pt"
    shutil.copy2(best_pt, dst_pt)
    print(f"  PT     → {dst_pt}")
    print("───────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
#  Validate dataset exists
# ─────────────────────────────────────────────────────────────────────────────

def _check_dataset():
    train_img = YOLO_DIR / "dataset" / "images" / "train"
    if not train_img.exists() or not any(train_img.iterdir()):
        print("⚠ Training images not found.")
        print("  Run first:  python ML_Model/yolo/prepare_dataset.py")
        return False
    n = sum(1 for _ in train_img.glob("*.jpg")) + sum(1 for _ in train_img.glob("*.png"))
    print(f"  Dataset OK — {n} training images found.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for BlindSpotGuard")
    parser.add_argument("--model",  default="yolov8n.pt",
                        help="Base model (yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=TRAIN_ARGS["epochs"])
    parser.add_argument("--imgsz",  type=int, default=TRAIN_ARGS["imgsz"])
    parser.add_argument("--batch",  type=int, default=TRAIN_ARGS["batch"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--quick",  action="store_true",
                        help="5-epoch quick smoke test (no export)")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, only export best.pt")
    args = parser.parse_args()

    # ── Dataset check ─────────────────────────────────────────────────────────
    if not args.export_only and not _check_dataset():
        sys.exit(1)

    if args.quick:
        args.epochs = 5
        TRAIN_ARGS["cache"] = False
        print("Quick mode: 5 epochs, no cache, no export.")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.resume:
        last_ckpt = Path(RUNS_DIR) / "blindspot_yolo" / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"⚠ No checkpoint found at {last_ckpt}")
            sys.exit(1)
        model = YOLO(str(last_ckpt))
        print(f"Resuming from: {last_ckpt}")
    else:
        model = YOLO(args.model)
        print(f"Starting from pretrained: {args.model}")

    print(f"Model: {model.info()}")

    if args.export_only:
        run_dir = Path(RUNS_DIR) / "blindspot_yolo"
        export_models(run_dir)
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    train_kwargs = {**TRAIN_ARGS, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch}
    if args.resume:
        train_kwargs["resume"] = True

    print(f"\nTraining for {args.epochs} epochs  |  imgsz={args.imgsz}  |  batch={args.batch}")
    print(f"Device: {train_kwargs['device']}   AMP: {train_kwargs['amp']}")
    print("─" * 60)

    results = model.train(**train_kwargs)

    # ── Post-training validation ───────────────────────────────────────────────
    run_dir = Path(RUNS_DIR) / "blindspot_yolo"
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        print("\n── Final Validation ──────────────────────────────────────")
        val_model = YOLO(str(best_pt))
        metrics   = val_model.val(data=DATASET_CFG, imgsz=args.imgsz, split="test")
        print(f"  mAP@0.5    : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        for cls_id, name in enumerate(["car","truck","motorcycle","bus","person","bicycle"]):
            if cls_id < len(metrics.box.ap):
                print(f"  {name:12s}: AP@0.5={metrics.box.ap50[cls_id]:.4f}")

    # ── Export ────────────────────────────────────────────────────────────────
    if not args.quick:
        export_models(run_dir)

    print(f"\n✅  Training complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()

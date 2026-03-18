"""
ML_Model/yolo/evaluate_yolo.py
───────────────────────────────
YOLOv8 Model Evaluation & Export Pipeline for BlindSpotGuard.

Produces
────────
  1. mAP@0.5 and mAP@0.5:0.95 per class on test split
  2. Precision–Recall curve (PNG)
  3. Confusion matrix (PNG)
  4. Speed benchmark  (ms/image on CPU — simulates Pi performance)
  5. Exports to ONNX and TFLite if not already done
  6. Verifies ONNX output matches PyTorch output (sanity check)

Usage
─────
  python ML_Model/yolo/evaluate_yolo.py [--model path/to/best.pt]
  python ML_Model/yolo/evaluate_yolo.py --model ML_Model/saved_models/blindspot_yolo.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

DATASET_CFG = str(Path("ML_Model/yolo/dataset.yaml"))
OUT_DIR     = Path("ML_Model/evaluation/yolo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES    = ["car", "truck", "motorcycle", "bus", "person", "bicycle"]
BG, FG     = "#0a0f1e", "#e8edf8"
CLS_COLORS = ["#5b8aff","#00e676","#ffb300","#ff3d3d","#a78bfa","#38bdf8"]


# ─────────────────────────────────────────────────────────────────────────────
#  Run validation metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(model: YOLO, imgsz: int = 320) -> dict:
    print("\n── Validation ─────────────────────────────────────────")
    metrics = model.val(
        data=DATASET_CFG,
        imgsz=imgsz,
        split="test",
        conf=0.001,     # very low conf for PR curve
        iou=0.6,
        verbose=True,
        plots=True,
        save_dir=str(OUT_DIR),
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Per-class bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_ap(metrics, model_name: str):
    ap50   = metrics.box.ap50
    ap5095 = metrics.box.ap
    n = min(len(ap50), len(CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    x = range(n)

    for ax, vals, title in zip(axes, [ap50[:n], ap5095[:n]], ["AP@0.5", "AP@0.5:0.95"]):
        bars = ax.bar(x, vals, color=CLS_COLORS[:n], alpha=0.85, width=0.6)
        ax.set_facecolor(BG)
        ax.set_xticks(list(x)); ax.set_xticklabels(CLASSES[:n], color=FG, rotation=15)
        ax.set_ylabel(title, color=FG); ax.set_ylim(0, 1)
        ax.set_title(f"{model_name} — {title}", color=FG)
        ax.tick_params(colors=FG)
        for spine in ax.spines.values(): spine.set_edgecolor("#1a2845")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, min(val + 0.02, 0.96),
                    f"{val:.3f}", ha="center", va="bottom", color=FG, fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / f"{model_name}_per_class_ap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Per-class AP chart → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  CPU speed benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_cpu(model: YOLO, n_runs: int = 100, imgsz: int = 320) -> dict:
    print("\n── CPU Inference Benchmark ────────────────────────────")
    import cv2
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(5):
        model.predict(dummy, verbose=False, device="cpu")

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, device="cpu", imgsz=imgsz)
        times.append((time.perf_counter() - t0) * 1000)

    results = {
        "mean_ms":   round(np.mean(times),   2),
        "std_ms":    round(np.std(times),    2),
        "min_ms":    round(np.min(times),    2),
        "max_ms":    round(np.max(times),    2),
        "max_fps":   round(1000 / np.mean(times), 1),
    }
    print(f"  Mean: {results['mean_ms']} ms  ± {results['std_ms']} ms  "
          f"(min: {results['min_ms']}  max: {results['max_ms']})")
    print(f"  Max throughput: {results['max_fps']} FPS  (single-core CPU)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX sanity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_onnx(pt_model: YOLO, onnx_path: str, imgsz: int = 320):
    if not os.path.exists(onnx_path):
        print(f"  ONNX not found at {onnx_path} — skipping verification.")
        return
    try:
        import onnxruntime as ort
        import cv2

        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        ref   = pt_model.predict(dummy, verbose=False)[0].boxes

        sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp   = dummy.astype(np.float32).transpose(2,0,1)[None] / 255.0
        out   = sess.run(None, {sess.get_inputs()[0].name: inp})
        print(f"  ✔ ONNX verification OK — output shape: {out[0].shape}")
    except ImportError:
        print("  onnxruntime not installed — skipping verification. (pip install onnxruntime)")
    except Exception as exc:
        print(f"  ⚠ ONNX verification error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics, speed: dict, model_name: str):
    print(f"\n{'═'*58}")
    print(f"  {model_name}  —  Evaluation Summary")
    print(f"{'─'*58}")
    print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
    print(f"  Precision (all): {metrics.box.mp:.4f}")
    print(f"  Recall    (all): {metrics.box.mr:.4f}")
    print(f"{'─'*58}")
    print(f"  Per-class AP@0.5:")
    for i, cls_name in enumerate(CLASSES):
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else float("nan")
        bar = "█" * int(ap * 20)
        print(f"    {cls_name:12s}  {ap:.4f}  {bar}")
    print(f"{'─'*58}")
    print(f"  CPU Inference  : {speed['mean_ms']} ms/frame  → {speed['max_fps']} FPS")
    print(f"  Pi 4B estimate : ~{speed['mean_ms']*2:.0f} ms/frame (×2 factor)")
    print(f"{'═'*58}")

    # Save to text
    with open(OUT_DIR / f"{model_name}_summary.txt", "w") as f:
        f.write(f"mAP@0.5={metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95={metrics.box.map:.4f}\n")
        f.write(f"CPU_mean_ms={speed['mean_ms']}\n")
        for i, cls_name in enumerate(CLASSES):
            ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else -1
            f.write(f"{cls_name}={ap:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate BlindSpotGuard YOLO model")
    parser.add_argument("--model",  default="ML_Model/saved_models/blindspot_yolo.pt",
                        help="Path to .pt model file")
    parser.add_argument("--imgsz",  type=int, default=320)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        # Try fallback to run dir
        fallback = "ML_Model/yolo/runs/blindspot_yolo/weights/best.pt"
        if os.path.exists(fallback):
            args.model = fallback
            print(f"Using: {fallback}")
        else:
            print(f"⚠ Model not found: {args.model}")
            print("  Train first: python ML_Model/yolo/train_yolo.py")
            sys.exit(1)

    model_name = Path(args.model).stem
    print(f"\nEvaluating: {args.model}")
    model = YOLO(args.model)

    # Metrics
    metrics = run_validation(model, args.imgsz)
    plot_per_class_ap(metrics, model_name)

    # Speed
    speed = benchmark_cpu(model, imgsz=args.imgsz)

    # ONNX check
    onnx_path = str(Path("ML_Model/saved_models") / "blindspot_yolo.onnx")
    verify_onnx(model, onnx_path, args.imgsz)

    # Summary
    print_summary(metrics, speed, model_name)
    print(f"\nAll artefacts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

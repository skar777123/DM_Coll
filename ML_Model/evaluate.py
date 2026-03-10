"""
ML_Model/evaluate.py
─────────────────────
Comprehensive evaluation script for both BlindSpotGuard ML models.

Generates:
  • Classification Report  (precision / recall / F1 per class)
  • Confusion Matrix        (saved as PNG)
  • ROC curves              (saved as PNG)
  • TTC Regression Metrics  (MAE, RMSE, R²)
  • Inference Latency       (mean ms/sample on CPU — for Pi estimation)
  • Training curve plots    (loss + accuracy vs epoch)

Run:
    python ML_Model/evaluate.py [--model lstm|fusion|both]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (headless Pi)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, mean_absolute_error, r2_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ML_Model.config_ml import PATHS, DATA, CLASSES
from ML_Model.data.dataset import load_datasets, make_loaders
from ML_Model.models.threat_lstm import ThreatLSTM
from ML_Model.models.fusion_net  import FusionNet

DEVICE = torch.device("cpu")   # evaluate on CPU (mirrors Pi)
OUT_DIR = "ML_Model/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"safe": "#00e676", "caution": "#ffb300", "critical": "#ff3d3d"}
BG     = "#0a0f1e"
FG     = "#e8edf8"


# ─────────────────────────────────────────────────────────────────────────────
#  Collect predictions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, is_sequence: bool):
    model.eval()
    all_probs, all_preds, all_true, all_ttc_pred, all_ttc_true = [], [], [], [], []

    for X, y_cls, y_ttc in loader:
        X = X.to(DEVICE)
        if not is_sequence:
            X = X[:, -1, :]
        logits, ttc_pred = model(X)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(y_cls.numpy())
        all_ttc_pred.append(ttc_pred.squeeze(1).cpu().numpy())
        all_ttc_true.append(y_ttc.numpy())

    return (
        np.vstack(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_true),
        np.concatenate(all_ttc_pred),
        np.concatenate(all_ttc_true),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color=FG, fontsize=9)
    ax.set_ylabel(ylabel, color=FG, fontsize=9)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2845")


def plot_confusion(y_true, y_pred, model_name: str):
    cm  = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    for ax, data, title in zip(axes, [cm, cm_norm], ["Counts", "Normalised"]):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, color=FG)
        ax.set_yticklabels(CLASSES, color=FG)
        ax.set_xlabel("Predicted", color=FG); ax.set_ylabel("True", color=FG)
        ax.set_title(f"{model_name} — Confusion Matrix ({title})", color=FG)
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        for spine in ax.spines.values(): spine.set_edgecolor("#1a2845")
        for i in range(3):
            for j in range(3):
                val = f"{data[i,j]:.2f}" if title == "Normalised" else str(data[i,j])
                ax.text(j, i, val, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{model_name}_confusion.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Confusion matrix → {out}")


def plot_roc(y_true, probs, model_name: str):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    _style_ax(ax, f"{model_name} — ROC Curves", "False Positive Rate", "True Positive Rate")

    y_bin = np.eye(3)[y_true]
    cls_colors = list(COLORS.values())
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cls_colors[i], lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0,1],[0,1], "--", color="#333d5c", lw=1)
    ax.legend(facecolor=BG, edgecolor="#1a2845", labelcolor=FG)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{model_name}_roc.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ROC curves       → {out}")


def plot_ttc(y_true_ttc, y_pred_ttc, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)

    # Scatter
    ax = axes[0]
    _style_ax(ax, f"{model_name} — TTC Prediction", "True TTC (s)", "Predicted TTC (s)")
    clip = 15.0
    mask = y_true_ttc < clip
    ax.scatter(y_true_ttc[mask], y_pred_ttc[mask], alpha=0.25, s=8, color="#5b8aff")
    lim = [0, clip]; ax.plot(lim, lim, "--", color="#ff3d3d", lw=1.5)
    ax.set_xlim(lim); ax.set_ylim(lim)

    # Error distribution
    ax2 = axes[1]
    _style_ax(ax2, f"{model_name} — TTC Error Distribution", "Error (s)", "Count")
    err = y_pred_ttc[mask] - y_true_ttc[mask]
    ax2.hist(err, bins=40, color="#5b8aff", alpha=0.8, edgecolor="none")

    mae  = mean_absolute_error(y_true_ttc[mask], y_pred_ttc[mask])
    rmse = np.sqrt(np.mean(err**2))
    r2   = r2_score(y_true_ttc[mask], y_pred_ttc[mask])
    ax2.axvline(0, color="#ff3d3d", lw=1.5)
    ax2.text(0.97, 0.97, f"MAE={mae:.2f}s\nRMSE={rmse:.2f}s\nR²={r2:.3f}",
             transform=ax2.transAxes, va="top", ha="right", color=FG, fontsize=9,
             bbox=dict(facecolor="#0d1424", edgecolor="#1a2845", boxstyle="round,pad=0.4"))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{model_name}_ttc.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  TTC scatter      → {out}")
    return mae, rmse, r2


def plot_training_curves(log_csv: str, model_name: str):
    import csv as _csv
    if not os.path.exists(log_csv):
        return
    rows = []
    with open(log_csv) as f:
        for row in _csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    if not rows:
        return

    epochs    = [r["epoch"] for r in rows]
    tr_loss   = [r["train_loss"] for r in rows]
    val_loss  = [r["val_loss"]   for r in rows]
    val_acc   = [r["val_acc"]    for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    _style_ax(axes[0], f"{model_name} — Loss", "Epoch", "Loss")
    axes[0].plot(epochs, tr_loss,  color="#5b8aff", lw=2, label="Train")
    axes[0].plot(epochs, val_loss, color="#00e676", lw=2, label="Val")
    axes[0].legend(facecolor=BG, edgecolor="#1a2845", labelcolor=FG)

    _style_ax(axes[1], f"{model_name} — Val Accuracy", "Epoch", "Accuracy")
    axes[1].plot(epochs, val_acc, color="#ffb300", lw=2)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{model_name}_curves.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Training curves  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Latency benchmark
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_latency(model, is_sequence: bool, n_runs: int = 200) -> float:
    model.eval()
    from ML_Model.config_ml import SEQUENCE_LEN, INPUT_FEATURES
    if is_sequence:
        dummy = torch.randn(1, SEQUENCE_LEN, INPUT_FEATURES)
    else:
        dummy = torch.randn(1, INPUT_FEATURES)

    # Warm-up
    for _ in range(20):
        model(dummy)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000   # ms
    return round(elapsed, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate one model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, ckpt_path: str, model_name: str, test_loader, is_sequence: bool):
    if not os.path.exists(ckpt_path):
        print(f"  ⚠ Checkpoint not found: {ckpt_path}  (train first)")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n{'═'*60}")
    print(f"  {model_name}  — epoch {ckpt['epoch']}  |  val_loss={ckpt['val_loss']:.4f}  |  val_acc={ckpt['val_acc']:.4f}")
    print(f"  Parameters : {model.count_parameters():,}")
    print(f"{'─'*60}")

    probs, preds, true, ttc_pred, ttc_true = collect_predictions(model, test_loader, is_sequence)

    # Classification report
    report = classification_report(true, preds, target_names=CLASSES, digits=4)
    print(report)

    # Save report
    with open(os.path.join(OUT_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(f"{model_name}\n{'─'*40}\n{report}\n")

    # Plots
    plot_confusion(true, preds, model_name)
    plot_roc(true, probs, model_name)
    mae, rmse, r2 = plot_ttc(ttc_true, ttc_pred, model_name)
    log_csv = os.path.join(PATHS["log_dir"], f"{model_name}_log.csv")
    plot_training_curves(log_csv, model_name)

    # Latency
    lat = benchmark_latency(model, is_sequence)
    print(f"  TTC  →  MAE={mae:.3f}s  RMSE={rmse:.3f}s  R²={r2:.4f}")
    print(f"  Latency (CPU, 1 sample): {lat} ms   →  max {1000/lat:.0f} Hz")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm","fusion","both"], default="both")
    parser.add_argument("--data",  default=DATA["output_path"])
    args = parser.parse_args()

    _, _, test_ds, _ = load_datasets(args.data, fit_scaler=False)
    _, _, test_loader = make_loaders(
        test_ds, test_ds, test_ds   # dummy train/val; only test_loader used
    )

    if args.model in ("lstm", "both"):
        evaluate(ThreatLSTM().to(DEVICE), PATHS["lstm_model"],   "ThreatLSTM",  test_loader, True)
    if args.model in ("fusion", "both"):
        evaluate(FusionNet().to(DEVICE),  PATHS["fusion_model"],  "FusionNet",   test_loader, False)

    print(f"\nAll evaluation artefacts saved to:  {OUT_DIR}/")


if __name__ == "__main__":
    main()

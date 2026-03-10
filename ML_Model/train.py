"""
ML_Model/train.py
─────────────────
Training script for BlindSpotGuard ML models.

Trains both models:
  1. ThreatLSTM  — sequence-based threat predictor (primary)
  2. FusionNet   — single-frame fast classifier (secondary / fallback)

Run:
    python ML_Model/train.py [--model lstm|fusion|both] [--epochs N]

Features:
  • Cosine annealing LR scheduler
  • Early stopping with patience
  • Gradient clipping
  • Per-epoch tensorboard-compatible log (CSV fallback)
  • Best model checkpoint saved automatically
  • Prints final training summary table
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ML_Model.config_ml import TRAIN, PATHS, DATA
from ML_Model.data.dataset import load_datasets, make_loaders
from ML_Model.models.threat_lstm import ThreatLSTM
from ML_Model.models.fusion_net  import FusionNet

# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
#  One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model,
    loader,
    optimizer=None,
    class_weights=None,
    is_sequence: bool = True,
):
    """Run one train or eval epoch. Returns dict of averaged metrics."""
    training = optimizer is not None
    model.train(training)

    total_loss = total_cls = total_ttc = total_acc = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(training):
        for X, y_cls, y_ttc in loader:
            X, y_cls, y_ttc = X.to(DEVICE), y_cls.to(DEVICE), y_ttc.to(DEVICE)

            # FusionNet takes only the last timestep
            if not is_sequence:
                X = X[:, -1, :]

            logits, ttc_pred = model(X)
            loss, cls_loss, ttc_loss = model.compute_loss(
                logits, ttc_pred, y_cls, y_ttc, class_weights
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
                optimizer.step()

            total_loss += loss.item()
            total_cls  += cls_loss.item()
            total_ttc  += ttc_loss.item()
            total_acc  += accuracy(logits, y_cls)
            n_batches  += 1

    return {
        "loss": total_loss / n_batches,
        "cls":  total_cls  / n_batches,
        "ttc":  total_ttc  / n_batches,
        "acc":  total_acc  / n_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    model_name: str,
    save_path:  str,
    train_loader,
    val_loader,
    class_weights,
    epochs: int,
    is_sequence: bool,
):
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN["learning_rate"],
        weight_decay=TRAIN["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    log_path = os.path.join(PATHS["log_dir"], f"{model_name}_log.csv")
    os.makedirs(PATHS["log_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n{'─'*65}")
    print(f"  Training  : {model_name}   ({model.count_parameters():,} params)  →  {DEVICE}")
    print(f"  Epochs    : {epochs}   |  Batch: {TRAIN['batch_size']}   |  LR: {TRAIN['learning_rate']}")
    print(f"{'─'*65}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val Acc':>8}  {'TTC MSE':>8}  {'LR':>8}")
    print(f"{'─'*65}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss","val_acc","val_ttc","lr"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr  = run_epoch(model, train_loader, optimizer, class_weights, is_sequence)
        val = run_epoch(model, val_loader,   None,      class_weights, is_sequence)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"  {epoch:>5}  {tr['loss']:>10.4f}  {val['loss']:>9.4f}  "
            f"{val['acc']:>7.3f}  {val['ttc']:>8.4f}  {lr:>8.2e}  "
            f"({'↓ best' if val['loss'] < best_val_loss else '      '}) {elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr["loss"], val["loss"], val["acc"], val["ttc"], lr])

        history.append({**{"epoch": epoch, "lr": lr}, **{f"tr_{k}": v for k,v in tr.items()},
                        **{f"val_{k}": v for k,v in val.items()}})

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            patience_ctr  = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc":  val["acc"],
            }, save_path)
            print(f"         ✔ Checkpoint saved → {save_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= TRAIN["early_stop_patience"]:
                print(f"\n  Early stopping at epoch {epoch} (patience={TRAIN['early_stop_patience']})")
                break

    print(f"{'─'*65}")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Log saved     : {log_path}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BlindSpotGuard ML models")
    parser.add_argument("--model",  choices=["lstm", "fusion", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    parser.add_argument("--data",   default=DATA["output_path"])
    args = parser.parse_args()

    # ── Generate data if not present ──────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"Dataset not found at {args.data}. Generating…")
        from ML_Model.data.generate_data import generate_dataset
        generate_dataset(output_path=args.data)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading dataset…")
    train_ds, val_ds, test_ds, scaler = load_datasets(args.data, fit_scaler=True)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    # Class weights (inverse frequency)
    labels  = train_ds.y.numpy()
    counts  = np.bincount(labels, minlength=3).astype(np.float32)
    cw      = torch.tensor(1.0 / (counts + 1), dtype=torch.float32).to(DEVICE)

    # ── Train LSTM ────────────────────────────────────────────────────────────
    if args.model in ("lstm", "both"):
        lstm = ThreatLSTM().to(DEVICE)
        train_model(
            lstm, "ThreatLSTM", PATHS["lstm_model"],
            train_loader, val_loader, cw, args.epochs, is_sequence=True,
        )

    # ── Train FusionNet ───────────────────────────────────────────────────────
    if args.model in ("fusion", "both"):
        fnet = FusionNet().to(DEVICE)
        train_model(
            fnet, "FusionNet", PATHS["fusion_model"],
            train_loader, val_loader, cw, args.epochs, is_sequence=False,
        )

    print("\n✅  Training complete. Run  python ML_Model/evaluate.py  to see full metrics.")


if __name__ == "__main__":
    main()

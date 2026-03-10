"""
ML_Model/data/dataset.py
─────────────────────────
PyTorch Dataset and DataLoader utilities for the BlindSpotGuard
threat predictor. Loads the pre-generated .npz file and provides
stratified train / val / test splits.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

from ML_Model.config_ml import TRAIN, PATHS


# ─────────────────────────────────────────────────────────────────────────────

class BlindSpotDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic sequences.

    Args:
        X   : float32 array [N, T, F]  — feature sequences
        y   : int64   array [N]         — class labels (0=safe,1=caution,2=critical)
        ttc : float32 array [N]         — time-to-collision (s)
    """

    def __init__(
        self,
        X:   np.ndarray,
        y:   np.ndarray,
        ttc: np.ndarray,
    ) -> None:
        self.X   = torch.tensor(X,   dtype=torch.float32)
        self.y   = torch.tensor(y,   dtype=torch.long)
        self.ttc = torch.tensor(ttc, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ttc[idx]


# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(
    npz_path: str,
    fit_scaler: bool = True,
) -> tuple[BlindSpotDataset, BlindSpotDataset, BlindSpotDataset, StandardScaler]:
    """
    Load .npz, split into train/val/test, fit (or load) a standard scaler,
    and return three Dataset objects plus the fitted scaler.

    Returns: (train_ds, val_ds, test_ds, scaler)
    """
    data = np.load(npz_path, allow_pickle=True)
    X   = data["X"].astype(np.float32)   # [N, T, F]
    y   = data["y"].astype(np.int64)
    ttc = data["ttc"].astype(np.float32)

    N, T, F = X.shape

    # ── Train / temp split, then val / test from temp ─────────────────────
    val_ratio  = TRAIN["val_split"]
    test_ratio = TRAIN["test_split"]

    X_train, X_temp, y_train, y_temp, ttc_train, ttc_temp = train_test_split(
        X, y, ttc,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=42,
    )
    X_val, X_test, y_val, y_test, ttc_val, ttc_test = train_test_split(
        X_temp, y_temp, ttc_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=y_temp,
        random_state=42,
    )

    # ── Feature scaling (fit on train, transform all) ──────────────────────
    scaler = StandardScaler()
    if fit_scaler:
        X_train_2d = X_train.reshape(-1, F)
        scaler.fit(X_train_2d)
        os.makedirs(os.path.dirname(PATHS["scaler"]), exist_ok=True)
        joblib.dump(scaler, PATHS["scaler"])
    else:
        scaler = joblib.load(PATHS["scaler"])

    def transform(arr):
        shape = arr.shape
        return scaler.transform(arr.reshape(-1, F)).reshape(shape).astype(np.float32)

    X_train = transform(X_train)
    X_val   = transform(X_val)
    X_test  = transform(X_test)

    print(f"Train: {len(y_train):5d}  |  Val: {len(y_val):5d}  |  Test: {len(y_test):5d}")

    return (
        BlindSpotDataset(X_train, y_train, ttc_train),
        BlindSpotDataset(X_val,   y_val,   ttc_val),
        BlindSpotDataset(X_test,  y_test,  ttc_test),
        scaler,
    )


def make_loaders(
    train_ds: BlindSpotDataset,
    val_ds:   BlindSpotDataset,
    test_ds:  BlindSpotDataset,
    batch_size: int = TRAIN["batch_size"],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders. Train loader uses WeightedRandomSampler
    for class-balanced batches (handles class imbalance automatically).
    """
    # Class weights for balanced sampling
    labels  = train_ds.y.numpy()
    counts  = np.bincount(labels, minlength=3)
    weights = 1.0 / (counts + 1e-6)
    sample_w = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader

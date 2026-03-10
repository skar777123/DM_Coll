"""
tests/test_ml_models.py
────────────────────────
Unit tests for ThreatLSTM, FusionNet, and MLThreatEngine.
All tests run without GPU or real hardware.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ML_Model.config_ml  import SEQUENCE_LEN, INPUT_FEATURES, NUM_CLASSES, N_SENSORS, SENSOR_FEATURES
from ML_Model.models.threat_lstm import ThreatLSTM
from ML_Model.models.fusion_net  import FusionNet

BATCH = 8


# ─────────────────────────────────────────────────────────────────────────────
#  ThreatLSTM
# ─────────────────────────────────────────────────────────────────────────────

def test_lstm_forward_shape():
    model = ThreatLSTM()
    x = torch.randn(BATCH, SEQUENCE_LEN, INPUT_FEATURES)
    logits, ttc = model(x)
    assert logits.shape == (BATCH, NUM_CLASSES), f"Bad logits shape: {logits.shape}"
    assert ttc.shape    == (BATCH, 1),           f"Bad TTC shape: {ttc.shape}"
    print(f"  ✔ LSTM forward: logits={logits.shape}  ttc={ttc.shape}")


def test_lstm_loss_runs():
    model   = ThreatLSTM()
    x       = torch.randn(BATCH, SEQUENCE_LEN, INPUT_FEATURES)
    y_cls   = torch.randint(0, NUM_CLASSES, (BATCH,))
    y_ttc   = torch.rand(BATCH) * 10
    logits, ttc = model(x)
    loss, cls_l, ttc_l = model.compute_loss(logits, ttc, y_cls, y_ttc)
    assert loss.item() > 0
    print(f"  ✔ LSTM loss: total={loss.item():.4f}  cls={cls_l.item():.4f}  ttc={ttc_l.item():.4f}")


def test_lstm_predict_probs_sum_to_one():
    model = ThreatLSTM()
    model.eval()
    x     = torch.randn(BATCH, SEQUENCE_LEN, INPUT_FEATURES)
    with torch.no_grad():
        probs, ttc = model.predict(x)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), f"Probs don't sum to 1: {sums}"
    assert (ttc >= 0).all(), "TTC must be non-negative"
    print(f"  ✔ LSTM probs sum to 1  |  TTC range: [{ttc.min():.2f}, {ttc.max():.2f}]")


def test_lstm_zone_labels():
    model  = ThreatLSTM()
    probs  = torch.tensor([[0.9, 0.08, 0.02],   # safe
                            [0.1, 0.85, 0.05],   # caution
                            [0.05, 0.1, 0.85]])  # critical
    zones  = model.zone_from_probs(probs)
    assert zones[0] == "safe",     f"Expected safe, got {zones[0]}"
    assert zones[1] in ("caution", "critical"), f"Unexpected: {zones[1]}"
    assert zones[2] == "critical", f"Expected critical, got {zones[2]}"
    print(f"  ✔ Zone labels: {zones}")


def test_lstm_parameter_count():
    model = ThreatLSTM()
    n = model.count_parameters()
    assert 50_000 < n < 2_000_000, f"Unexpected param count: {n:,}"
    print(f"  ✔ LSTM parameters: {n:,}")


# ─────────────────────────────────────────────────────────────────────────────
#  FusionNet
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_forward_shape():
    model = FusionNet()
    model.eval()
    x = torch.randn(BATCH, INPUT_FEATURES)
    logits, ttc = model(x)
    assert logits.shape == (BATCH, NUM_CLASSES)
    assert ttc.shape    == (BATCH, 1)
    print(f"  ✔ FusionNet forward: logits={logits.shape}  ttc={ttc.shape}")


def test_fusion_predict():
    model = FusionNet()
    model.eval()
    x = torch.randn(BATCH, INPUT_FEATURES)
    with torch.no_grad():
        probs, ttc = model.predict(x)
    assert probs.shape == (BATCH, NUM_CLASSES)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)
    print(f"  ✔ FusionNet predict: probs sum={sums.mean():.6f}")


def test_fusion_loss():
    model  = FusionNet()
    x      = torch.randn(BATCH, INPUT_FEATURES)
    y_cls  = torch.zeros(BATCH, dtype=torch.long)
    y_ttc  = torch.zeros(BATCH)
    logits, ttc = model(x)
    loss, cls_l, ttc_l = model.compute_loss(logits, ttc, y_cls, y_ttc)
    assert loss.item() > 0
    print(f"  ✔ FusionNet loss: {loss.item():.4f}")


def test_fusion_lightweight():
    model = FusionNet()
    n = model.count_parameters()
    assert n < 200_000, f"FusionNet too large for Pi: {n:,} params"
    print(f"  ✔ FusionNet parameters: {n:,}  (Pi-safe)")


# ─────────────────────────────────────────────────────────────────────────────
#  Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_data_generation_small():
    """Generate 50 samples and verify shapes."""
    from ML_Model.data.generate_data import generate_dataset
    out = "/tmp/test_dataset.npz"
    generate_dataset(n_samples=50, output_path=out)
    data = np.load(out)
    assert data["X"].shape == (50, SEQUENCE_LEN, INPUT_FEATURES), f"Bad X shape: {data['X'].shape}"
    assert data["y"].shape == (50,)
    assert set(data["y"]).issubset({0, 1, 2})
    print(f"  ✔ Data generation: X={data['X'].shape}  y={data['y'].shape}")
    os.remove(out)


def test_feature_engineering():
    """Snapshot features should be in expected ranges."""
    from ML_Model.inference import _build_snapshot
    dists = [200.0, 350.0, 60.0]
    vels  = [50.0, 0.0, 200.0]
    accs  = [10.0, 0.0, 100.0]
    cams  = [False, False, True]
    snap  = _build_snapshot(dists, vels, accs, cams)
    assert snap.shape == (INPUT_FEATURES,)
    # Distance features should be in [0,1]
    for i in range(N_SENSORS):
        d_norm = snap[i * SENSOR_FEATURES]
        assert 0 <= d_norm <= 1, f"Distance out of range: {d_norm}"
    d_norms = [round(float(snap[i * SENSOR_FEATURES]), 3) for i in range(N_SENSORS)]
    print(f"  ✔ Feature snapshot: shape={snap.shape}  d_norms={d_norms}")


# ─────────────────────────────────────────────────────────────────────────────
#  Latency smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_lstm_cpu_latency():
    import time
    model = ThreatLSTM(); model.eval()
    x = torch.randn(1, SEQUENCE_LEN, INPUT_FEATURES)
    with torch.no_grad():
        for _ in range(10): model(x)   # warm-up
        t0 = time.perf_counter()
        for _ in range(100): model(x)
        ms = (time.perf_counter() - t0) / 100 * 1000
    assert ms < 50, f"LSTM too slow: {ms:.2f} ms (target <50ms on PC)"
    print(f"  ✔ LSTM CPU latency: {ms:.2f} ms/sample")


def test_fusion_cpu_latency():
    import time
    model = FusionNet(); model.eval()
    x = torch.randn(1, INPUT_FEATURES)
    with torch.no_grad():
        for _ in range(10): model(x)
        t0 = time.perf_counter()
        for _ in range(500): model(x)
        ms = (time.perf_counter() - t0) / 500 * 1000
    assert ms < 5, f"FusionNet too slow: {ms:.2f} ms"
    print(f"  ✔ FusionNet CPU latency: {ms:.3f} ms/sample")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("━━ ThreatLSTM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    test_lstm_forward_shape()
    test_lstm_loss_runs()
    test_lstm_predict_probs_sum_to_one()
    test_lstm_zone_labels()
    test_lstm_parameter_count()
    test_lstm_cpu_latency()

    print("\n━━ FusionNet ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    test_fusion_forward_shape()
    test_fusion_predict()
    test_fusion_loss()
    test_fusion_lightweight()
    test_fusion_cpu_latency()

    print("\n━━ Data Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    test_data_generation_small()
    test_feature_engineering()

    print("\n✅  All ML tests passed.")

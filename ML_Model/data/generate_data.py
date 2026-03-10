"""
ML_Model/data/generate_data.py
───────────────────────────────
Synthetic Training Data Generator for BlindSpotGuard Threat Predictor.

Generates realistic sensor time-series sequences across 6 driving scenarios.
Each sequence = SEQUENCE_LEN timesteps of [dist, velocity, acceleration,
zone_encoded, cam_threat] for 3 sensors, labelled with [safe/caution/critical]
and time-to-collision (TTC).

Scenarios Modelled
──────────────────
  safe          — steady traffic, no approaching vehicle
  slow_approach — vehicle approaching at ≤ 30 km/h
  fast_approach — vehicle approaching at 60–120 km/h (cut-in / rear-end)
  cut_in        — rapid lateral approach into blindspot
  pass_by       — vehicle approaches then passes (threat then clears)
  rear_tailgate — aggressive tailgating scenario

Usage
─────
  python ML_Model/data/generate_data.py
  # Creates: ML_Model/data/synthetic_dataset.npz
"""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ML_Model.config_ml import DATA, SEQUENCE_LEN, SENSOR_FEATURES, N_SENSORS, CLASSES

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

DT = 0.02   # 20 ms timestep → 50 Hz

def _add_noise(arr: np.ndarray, std: float = DATA["noise_std_cm"]) -> np.ndarray:
    return arr + np.random.normal(0, std, arr.shape)

def _zone_encode(dist_cm: float) -> float:
    """Encode zone as 0/0.5/1.0 for neutral/caution/critical."""
    from config import ZONE
    if dist_cm <= ZONE["critical"]:  return 1.0
    if dist_cm <= ZONE["caution"]:   return 0.5
    return 0.0

def _velocity(distances: np.ndarray) -> np.ndarray:
    """Approximate velocity (cm/s) as finite difference."""
    vel = np.zeros_like(distances)
    vel[1:] = (distances[:-1] - distances[1:]) / DT   # positive = approaching
    vel[0]  = vel[1]
    return vel

def _acceleration(velocities: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(velocities)
    acc[1:] = (velocities[1:] - velocities[:-1]) / DT
    acc[0]  = acc[1]
    return acc

def _ttc(dist_cm: float, vel_cms: float) -> float:
    """Time-to-collision in seconds. inf if not approaching."""
    if vel_cms <= 0:
        return 999.0
    return max(0.0, dist_cm / vel_cms)


# ─────────────────────────────────────────────────────────────────────────────
#  Scenario Generators  (return distances array per sensor, shape [SEQUENCE_LEN])
# ─────────────────────────────────────────────────────────────────────────────

T = SEQUENCE_LEN

def _gen_safe() -> tuple[np.ndarray, int, float]:
    """All sensors read safe distances with slight fluctuation."""
    dists = np.zeros((T, N_SENSORS))
    for i in range(N_SENSORS):
        base = np.random.uniform(280, 400)
        drift = np.cumsum(np.random.normal(0, 0.3, T))
        dists[:, i] = np.clip(base + drift, 200, 400)
    return dists, 0, 999.0   # label=safe, TTC=inf


def _gen_slow_approach(sensor_idx: int = None) -> tuple[np.ndarray, int, float]:
    """One sensor sees a slow-approaching vehicle (30–60 cm/s)."""
    if sensor_idx is None:
        sensor_idx = np.random.randint(0, N_SENSORS)
    speed = np.random.uniform(30, 80)   # cm/s
    start_dist = np.random.uniform(250, 380)
    dists = np.zeros((T, N_SENSORS))

    # other sensors: safe
    for i in range(N_SENSORS):
        if i == sensor_idx:
            d = start_dist - speed * DT * np.arange(T)
            dists[:, i] = np.clip(d, 5, 400)
        else:
            base = np.random.uniform(280, 400)
            dists[:, i] = np.clip(base + np.random.normal(0, 1, T), 200, 400)

    final_dist = dists[-1, sensor_idx]
    label = 0 if final_dist > 200 else (2 if final_dist < 80 else 1)
    ttc   = _ttc(final_dist, speed)
    return dists, label, ttc


def _gen_fast_approach(sensor_idx: int = None) -> tuple[np.ndarray, int, float]:
    """One sensor sees a fast-approaching vehicle (fast closing speed)."""
    if sensor_idx is None:
        sensor_idx = np.random.randint(0, N_SENSORS)
    speed = np.random.uniform(120, 280)   # cm/s  ≈ 4–10 km/h in 0.6s window
    start_dist = np.random.uniform(180, 350)
    dists = np.zeros((T, N_SENSORS))

    for i in range(N_SENSORS):
        if i == sensor_idx:
            d = start_dist - speed * DT * np.arange(T)
            dists[:, i] = np.clip(d, 5, 400)
        else:
            base = np.random.uniform(280, 400)
            dists[:, i] = np.clip(base + np.random.normal(0, 1, T), 200, 400)

    final_dist = dists[-1, sensor_idx]
    label = 2  # almost always critical with this speed
    ttc   = _ttc(final_dist, speed)
    return dists, label, ttc


def _gen_cut_in() -> tuple[np.ndarray, int, float]:
    """Left or right sensor: sudden lateral approach (starts safe, drops fast)."""
    sensor_idx = np.random.choice([0, 1])   # left or right only
    dists = np.zeros((T, N_SENSORS))
    for i in range(N_SENSORS):
        dists[:, i] = np.random.uniform(300, 400)   # all safe to start

    # After halfway, sharp drop in blindspot sensor
    drop_start = T // 2
    speed = np.random.uniform(200, 400)
    start_dist = np.random.uniform(200, 300)

    for t in range(T):
        if t < drop_start:
            dists[t, sensor_idx] = start_dist + np.random.normal(0, 2)
        else:
            elapsed = t - drop_start
            dists[t, sensor_idx] = max(5, start_dist - speed * DT * elapsed)

    final_dist = dists[-1, sensor_idx]
    label = 2 if final_dist < 80 else 1
    ttc   = _ttc(final_dist, speed)
    return dists, label, ttc


def _gen_pass_by() -> tuple[np.ndarray, int, float]:
    """Vehicle approaches to caution zone then moves away."""
    sensor_idx = np.random.randint(0, N_SENSORS)
    dists = np.zeros((T, N_SENSORS))
    for i in range(N_SENSORS):
        if i != sensor_idx:
            dists[:, i] = np.random.uniform(300, 400)

    # Parabolic distance: approaches then recedes
    peak   = np.random.uniform(100, 160)    # closest point
    apex_t = np.random.randint(T // 3, 2 * T // 3)
    start  = np.random.uniform(300, 380)
    end    = np.random.uniform(200, 380)

    for t in range(T):
        x = t / T
        # Simple quadratic peak
        if t <= apex_t:
            alpha = t / apex_t
            dists[t, sensor_idx] = start + (peak - start) * alpha
        else:
            alpha = (t - apex_t) / (T - apex_t)
            dists[t, sensor_idx] = peak + (end - peak) * alpha

    final_dist = dists[-1, sensor_idx]
    label = 1 if final_dist < 200 else 0   # ends caution or safe
    return dists, label, 999.0


def _gen_rear_tailgate() -> tuple[np.ndarray, int, float]:
    """Rear sensor: aggressive tailgating — stays in critical zone."""
    dists = np.zeros((T, N_SENSORS))
    # Left and right: safe
    for i in [0, 1]:
        dists[:, i] = np.clip(np.random.uniform(280, 400) + np.random.normal(0, 1, T), 200, 400)
    # Rear: very close, oscillating
    base = np.random.uniform(30, 75)
    dists[:, 2] = np.clip(base + np.random.normal(0, 3, T), 10, 100)

    final_dist = dists[-1, 2]
    label = 2 if final_dist < 80 else 1
    ttc   = _ttc(final_dist, 50)   # assume some approach speed
    return dists, label, ttc


_SCENARIO_FN = {
    "safe":          _gen_safe,
    "slow_approach": _gen_slow_approach,
    "fast_approach": _gen_fast_approach,
    "cut_in":        _gen_cut_in,
    "pass_by":       _gen_pass_by,
    "rear_tailgate": _gen_rear_tailgate,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(dists: np.ndarray) -> np.ndarray:
    """
    Convert raw distance matrix [T, 3] → feature matrix [T, 15].
    Per sensor: [dist, velocity, acceleration, zone_encoded, cam_threat_sim]
    """
    from config import ZONE
    T_local, N = dists.shape
    feats = np.zeros((T_local, N * SENSOR_FEATURES))

    for i in range(N):
        d   = _add_noise(dists[:, i])
        vel = _velocity(d)
        acc = _acceleration(vel)
        zone_enc = np.array([_zone_encode(v) for v in d])
        # Simulated camera threat — 1 if approaching fast in caution zone
        cam_threat = ((vel > 50) & (zone_enc > 0)).astype(float)
        cam_threat += np.random.binomial(1, 0.05, T_local)   # occasional false positive

        col = i * SENSOR_FEATURES
        feats[:, col + 0] = d / 400.0            # normalise to [0,1]
        feats[:, col + 1] = np.clip(vel / 400.0, -1, 1)
        feats[:, col + 2] = np.clip(acc / 500.0, -1, 1)
        feats[:, col + 3] = zone_enc
        feats[:, col + 4] = np.clip(cam_threat, 0, 1)

    return feats.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_samples: int = DATA["n_samples"],
    output_path: str = DATA["output_path"],
) -> None:
    weights = DATA["scenario_weights"]
    scenarios = list(weights.keys())
    probs     = np.array([weights[s] for s in scenarios])
    probs     = probs / probs.sum()

    X_list, y_list, ttc_list, scenario_list = [], [], [], []

    print(f"Generating {n_samples} sequences…")
    for i in range(n_samples):
        sc_name = np.random.choice(scenarios, p=probs)
        fn      = _SCENARIO_FN[sc_name]
        dists, label, ttc = fn()

        feats = _build_features(dists)   # [T, 15]
        X_list.append(feats)
        y_list.append(label)
        ttc_list.append(min(ttc, 30.0))   # cap at 30 s
        scenario_list.append(scenarios.index(sc_name))

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_samples} done.")

    X   = np.stack(X_list)           # [N, T, 15]
    y   = np.array(y_list, dtype=np.int64)
    ttc = np.array(ttc_list, dtype=np.float32)
    sc  = np.array(scenario_list, dtype=np.int64)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path, X=X, y=y, ttc=ttc, scenarios=sc,
        class_names=np.array(CLASSES),
        scenario_names=np.array(scenarios),
    )

    print(f"\nDataset saved → {output_path}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print("Label distribution:")
    for ci, cn in enumerate(CLASSES):
        print(f"  {cn:12s}: {(y == ci).sum():5d}  ({100*(y==ci).mean():.1f}%)")


if __name__ == "__main__":
    generate_dataset()

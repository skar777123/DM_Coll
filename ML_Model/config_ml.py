"""
ML_Model/config_ml.py
─────────────────────
ML pipeline configuration for the BlindSpotGuard threat predictor.
"""

# ── Sequence / Feature Settings ───────────────────────────────────────────────
SEQUENCE_LEN   = 30        # Number of timesteps fed into LSTM (30 × 20ms = 0.6s)
SENSOR_FEATURES = 5        # Per sensor: [dist, velocity, acceleration, zone_enc, cam_threat]
N_SENSORS       = 3        # left, right, rear
INPUT_FEATURES  = SENSOR_FEATURES * N_SENSORS   # 15 total

# ── Model Architecture ────────────────────────────────────────────────────────
LSTM = {
    "hidden_size":  128,
    "num_layers":   2,
    "dropout":      0.3,
    "bidirectional": False,
}

FUSION_NET = {
    "hidden_dims": [128, 64, 32],
    "dropout":     0.3,
}

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN = {
    "epochs":          10,
    "batch_size":      64,
    "learning_rate":   1e-3,
    "weight_decay":    1e-4,
    "scheduler_step":  15,
    "scheduler_gamma": 0.5,
    "val_split":       0.15,
    "test_split":      0.10,
    "early_stop_patience": 10,
    "grad_clip":       1.0,
}

# ── Data Generation ───────────────────────────────────────────────────────────
DATA = {
    "n_samples":           12000,   # Total synthetic sequences
    "noise_std_cm":        2.5,     # Sensor noise (cm)
    "scenario_weights": {           # Proportion of each scenario in dataset
        "safe":            0.30,
        "slow_approach":   0.20,
        "fast_approach":   0.20,
        "cut_in":          0.12,
        "pass_by":         0.10,
        "rear_tailgate":   0.08,
    },
    "output_path": "ML_Model/data/synthetic_dataset.npz",
}

# ── Thresholds (post-model) ───────────────────────────────────────────────────
THRESHOLDS = {
    "critical_prob": 0.70,    # threat_prob ≥ this → critical
    "caution_prob":  0.35,    # threat_prob ≥ this → caution
    "ttc_critical":  1.5,     # TTC ≤ 1.5s → critical regardless of prob
    "ttc_caution":   3.0,     # TTC ≤ 3.0s → caution
}

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES        = ["safe", "caution", "critical"]
NUM_CLASSES    = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {
    "model_dir":    "ML_Model/saved_models",
    "lstm_model":   "ML_Model/saved_models/threat_lstm.pt",
    "fusion_model": "ML_Model/saved_models/fusion_net.pt",
    "yolo_model":   "ML_Model/saved_models/blindspot_yolo.pt",   # custom-trained YOLO
    "scaler":       "ML_Model/saved_models/scaler.pkl",
    "log_dir":      "ML_Model/runs",
}

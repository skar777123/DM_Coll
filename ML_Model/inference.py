"""
ML_Model/inference.py
──────────────────────
Real-Time ML Inference Engine for BlindSpotGuard.

Replaces the simple threshold-based ZoneEvaluator with ML predictions.
Integrates seamlessly with main.py — same interface, smarter decisions.

Model priority
──────────────
  1. ThreatLSTM   — uses a rolling 0.6-s window of sensor history.
                    Primary model, highest accuracy.
  2. FusionNet    — uses only the latest snapshot. Falls back when
                    the LSTM buffer isn't full yet.
  3. Custom YOLO  — loaded from ML_Model/saved_models/blindspot_yolo.pt
                    (if trained) or falls back to yolov8n.pt.

All models run on CPU (Pi-compatible). LSTM ≈ 2 ms, FusionNet ≈ 0.1 ms,
YOLO ≈ 60–150 ms (rate-limited to 5 Hz to avoid starving sensor loop).

Usage
─────
  # Drop-in replacement for ZoneEvaluator in main.py:
  from ML_Model.inference import MLThreatEngine
  engine = MLThreatEngine(leds, motors)
  state  = engine.evaluate(ultrasonic_data, camera_frames)
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from utils_safety import is_module_safe

# Lazy torch imports to avoid SIGILL on startup
torch = None
F = None

def _import_torch():
    global torch, F
    if torch is None:
        if is_module_safe("torch"):
            import torch as t
            import torch.nn.functional as f
            torch = t
            F = f
            return True
        return False
    return True

if TYPE_CHECKING:
    from alerts.leds   import LEDController
    from alerts.motors import MotorController

from detection.zone_logic import ZoneEvaluator, SystemState, DirectionState
from ML_Model.config_ml   import (
    SEQUENCE_LEN, INPUT_FEATURES, SENSOR_FEATURES, N_SENSORS,
    PATHS, THRESHOLDS, CLASSES,
)

log = logging.getLogger(__name__)

# DEVICE will be set after torch is imported
DEVICE = None
DT     = 0.02   # 20 ms (50 Hz)

# ─────────────────────────────────────────────────────────────────────────────
#  Feature builder (mirrors generate_data.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _build_snapshot(
    distances: list[float],   # [left, right, rear] latest cm
    velocities: list[float],  # cm/s (positive = approaching)
    accs: list[float],        # cm/s²
    cam_threats: list[bool],
) -> np.ndarray:
    """Build one [INPUT_FEATURES] feature vector from latest readings."""
    from config import ZONE
    feats = np.zeros(INPUT_FEATURES, dtype=np.float32)
    for i in range(N_SENSORS):
        d   = distances[i]
        vel = velocities[i]
        acc = accs[i]

        zone_enc = 1.0 if d <= ZONE["critical"] else (0.5 if d <= ZONE["caution"] else 0.0)
        cam      = float(cam_threats[i])

        col = i * SENSOR_FEATURES
        feats[col + 0] = min(d, 400) / 400.0
        feats[col + 1] = np.clip(vel / 400.0, -1, 1)
        feats[col + 2] = np.clip(acc / 500.0, -1, 1)
        feats[col + 3] = zone_enc
        feats[col + 4] = cam
    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  Rolling history for LSTM
# ─────────────────────────────────────────────────────────────────────────────

class _SensorHistory:
    """Maintains a rolling window of sensor readings for LSTM input."""

    def __init__(self) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LEN)
        self._prev_dists  = [300.0, 300.0, 300.0]
        self._prev_vels   = [0.0,   0.0,   0.0]

    def push(self, distances: list[float], cam_threats: list[bool]) -> None:
        vels = [(self._prev_dists[i] - distances[i]) / DT for i in range(N_SENSORS)]
        accs = [(vels[i] - self._prev_vels[i]) / DT      for i in range(N_SENSORS)]
        snap = _build_snapshot(distances, vels, accs, cam_threats)
        self._buffer.append(snap)
        self._prev_dists = distances[:]
        self._prev_vels  = vels[:]

    @property
    def ready(self) -> bool:
        return len(self._buffer) == SEQUENCE_LEN

    @property
    def latest_snap(self) -> np.ndarray:
        return self._buffer[-1] if self._buffer else np.zeros(INPUT_FEATURES, np.float32)

    def get_sequence(self) -> np.ndarray:
        return np.stack(self._buffer)   # [T, F]


# ─────────────────────────────────────────────────────────────────────────────
#  YOLO camera wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _YOLOInference:
    """Wraps YOLOv8 with rate-limiting (5 Hz) for Pi safety."""

    def __init__(self) -> None:
        self._model = None
        self._last  = time.time()
        self._rate  = 0.2   # min 200 ms between calls
        self._load()

    def _load(self) -> None:
        if not is_module_safe("ultralytics"):
            log.warning("YOLO (ultralytics) is unsafe or unavailable on this system — disabling YOLO.")
            return

        try:
            from ultralytics import YOLO
            custom = PATHS.get("yolo_model", "ML_Model/saved_models/blindspot_yolo.pt")
            if os.path.exists(custom):
                self._model = YOLO(custom)
                log.info("Custom YOLO loaded: %s", custom)
            else:
                self._model = YOLO("yolov8n.pt")
                log.info("Fallback to yolov8n.pt (custom model not found — train first)")
        except Exception as exc:
            log.warning("YOLO not available: %s", exc)

    def detect(self, frame_bytes: bytes | None) -> list[dict]:
        """Run detection on raw JPEG bytes. Returns list of objects."""
        if self._model is None or frame_bytes is None:
            return []
        now = time.time()
        if now - self._last < self._rate:
            return []   # rate-limited
        self._last = now

        try:
            import cv2
            arr    = np.frombuffer(frame_bytes, np.uint8)
            img    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return []
            results = self._model.predict(img, conf=0.35, verbose=False)
            dets    = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label  = r.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)
                    dets.append({"label": label, "conf": float(box.conf[0]), "area": area})
            return dets
        except Exception as exc:
            log.debug("YOLO detection error: %s", exc)
            return []


def _no_grad(func):
    def wrapper(*args, **kwargs):
        if torch is not None:
            with torch.no_grad():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

# ─────────────────────────────────────────────────────────────────────────────
#  Main engine
# ─────────────────────────────────────────────────────────────────────────────

class MLThreatEngine(ZoneEvaluator):
    """
    Drop-in replacement for ZoneEvaluator that uses ML models.

    Falls back to rule-based logic if models are unavailable.

    Usage (in main.py):
        from ML_Model.inference import MLThreatEngine
        engine = MLThreatEngine(leds=_led_ctrl, motors=_motor_ctrl)
        state  = engine.evaluate(ultrasonic_data, camera_frames)
    """

    def __init__(self, leds: "LEDController", motors: "MotorController") -> None:
        super().__init__(leds=leds, motors=motors)
        self._history = _SensorHistory()
        
        # Try to import torch before loading models
        global DEVICE
        if _import_torch():
            DEVICE = torch.device("cpu")
            self._scaler  = self._load_scaler()
            self._lstm    = self._load_model("lstm")
            self._fnet    = self._load_model("fusion")
        else:
            log.warning("PyTorch is unsafe or unavailable on this system — ML models disabled.")
            self._scaler = None
            self._lstm = None
            self._fnet = None

        self._yolo    = _YOLOInference()
        log.info("MLThreatEngine ready  (LSTM=%s  FusionNet=%s  YOLO=%s)",
                 self._lstm is not None, self._fnet is not None, self._yolo._model is not None)

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_scaler(self):
        try:
            import joblib
            if os.path.exists(PATHS["scaler"]):
                sc = joblib.load(PATHS["scaler"])
                log.info("Scaler loaded: %s", PATHS["scaler"])
                return sc
        except Exception as exc:
            log.warning("Scaler not loaded: %s", exc)
        return None

    def _load_model(self, model_type: str):
        path_key = "lstm_model" if model_type == "lstm" else "fusion_model"
        path = PATHS.get(path_key, "")
        if not os.path.exists(path):
            log.warning("%s model not found at %s — run train.py first.", model_type, path)
            return None
        try:
            if model_type == "lstm":
                from ML_Model.models.threat_lstm import ThreatLSTM
                m = ThreatLSTM().to(DEVICE)
            else:
                from ML_Model.models.fusion_net import FusionNet
                m = FusionNet().to(DEVICE)
            ckpt = torch.load(path, map_location=DEVICE)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            log.info("%s loaded from %s  (val_acc=%.4f)", model_type, path, ckpt.get("val_acc", -1))
            return m
        except Exception as exc:
            log.warning("Failed to load %s: %s", model_type, exc)
            return None

    # ── Scale features ────────────────────────────────────────────────────────

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return arr
        try:
            return self._scaler.transform(arr.reshape(1, -1)).reshape(arr.shape).astype(np.float32)
        except Exception:
            return arr

    # ── Override evaluate ─────────────────────────────────────────────────────

    def evaluate(
        self,
        ultrasonic_data: Dict[str, dict],
        camera_frames:   Dict,
    ) -> SystemState:
        """ML-enhanced evaluate — falls back to rule-based if models unavailable."""

        DIRS    = ("left", "right", "rear")
        dists   = [ultrasonic_data.get(d, {}).get("distance_cm", 300.0) for d in DIRS]
        cam_thr = [
            (camera_frames.get(d).threat if camera_frames.get(d) else False)
            for d in DIRS
        ]

        # Push to history
        self._history.push(dists, cam_thr)

        # Run YOLO detections on camera bytes
        yolo_results = {}
        for i, d in enumerate(DIRS):
            frame = camera_frames.get(d)
            if frame and frame.raw_jpeg:
                yolo_results[d] = self._yolo.detect(frame.raw_jpeg)
            else:
                yolo_results[d] = []

        # ── Try LSTM (primary) ────────────────────────────────────────────────
        ml_zones: Optional[list[str]] = None
        ml_ttcs:  Optional[list[float]] = None

        if self._lstm is not None and self._history.ready:
            ml_zones, ml_ttcs = self._lstm_predict()

        # ── Fallback to FusionNet ─────────────────────────────────────────────
        if ml_zones is None and self._fnet is not None:
            ml_zones, ml_ttcs = self._fnet_predict()

        # ── Fall all the way back to rule-based ───────────────────────────────
        if ml_zones is None:
            return super().evaluate(ultrasonic_data, camera_frames)

        # ── Merge ML zone with YOLO and hard safety constraints ───────────────
        final_zones = self._merge_zones(ml_zones, ml_ttcs, dists, yolo_results)

        # ── Build SystemState ─────────────────────────────────────────────────
        new_state = SystemState(timestamp=time.time())
        from config import ZONE

        for i, direction in enumerate(DIRS):
            dist = dists[i]
            frame = camera_frames.get(direction)
            is_v = frame.is_vehicle if frame else False
            is_m = frame.is_moving if frame else False

            # ── Apply USER FILTER: only alert if it is a MOVING VEHICLE within 200cm ──
            if is_v and is_m and dist <= ZONE["caution"]:
                zone = final_zones[i]
                # Double-check distance zone if ML didn't pick it up
                if dist <= ZONE["critical"]: zone = "critical"
                elif zone == "safe": zone = "caution"
            else:
                zone = "safe"

            # ── Apply Overrides ─────────────────────────────────────────────
            if direction in self._overrides:
                zone = self._overrides[direction]
                if zone == "critical": dist = min(dist, 50.0)
                elif zone == "caution": dist = min(dist, 150.0)
            
            led_mode   = {"safe": "off", "caution": "solid", "critical": "flash"}[zone]
            motor_mode = "pulse" if zone == "critical" else "off"
            
            ds = DirectionState(
                direction=direction,
                zone=zone,
                distance_cm=dist,
                camera_threat=bool(cam_thr[i]),
                is_vehicle=is_v,
                is_moving=is_m,
                led_mode=led_mode,
                motor_mode=motor_mode,
            )
            setattr(new_state, direction, ds)

        self._state = new_state
        self._apply_outputs(new_state)
        return new_state

    # ── LSTM prediction ───────────────────────────────────────────────────────

    @_no_grad
    def _lstm_predict(self) -> tuple[list[str], list[float]]:
        seq  = self._history.get_sequence()   # [T, F]
        seq  = self._scale_sequence(seq)
        x    = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        probs, ttc = self._lstm.predict(x)
        zones = self._lstm.zone_from_probs(probs)
        
        # Broadcast global prediction to all sensors if only one result is returned
        ttc_list = ttc.tolist()
        if len(zones) == 1:
            return zones * N_SENSORS, ttc_list * N_SENSORS
        return zones, ttc_list

    # Each direction shares the same sequence → the model predicts per-batch
    # But we pass one unified feature vector per timestep (all 3 sensors together).
    # So the output is a SINGLE zone for the most-threatening sensor.
    # We post-process: assign that threat to the most-distant sensor from safe.

    @_no_grad
    def _fnet_predict(self) -> tuple[list[str], list[float]]:
        snap  = self._history.latest_snap    # [INPUT_FEATURES]
        snap_s = self._scale(snap)
        x     = torch.tensor(snap_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        probs, ttc = self._fnet.predict(x)
        # FusionNet outputs one global zone — map to most dangerous sensor
        global_zone = self._fnet.zone_from_probs(probs)[0]
        # Override: use per-sensor distance thresholds for individual zones
        from config import ZONE
        per_dir = []
        ttcs    = []
        for snap_i in range(N_SENSORS):
            col  = snap_i * SENSOR_FEATURES
            snap_sensor = snap[col:col + SENSOR_FEATURES]
            x_s = torch.tensor(self._scale(snap_sensor), dtype=torch.float32)
            # Simple override logic
            d = snap_sensor[0] * 400.0   # de-normalize
            if d <= ZONE["critical"]:
                per_dir.append("critical")
            elif d <= ZONE["caution"]:
                per_dir.append("caution" if global_zone != "safe" else "caution")
            else:
                per_dir.append("safe")
            vel = snap_sensor[1] * 400.0
            ttcs.append(d / max(vel, 0.01) if vel > 0 else 999.0)
        return per_dir, ttcs

    # ── Scale sequence ────────────────────────────────────────────────────────

    def _scale_sequence(self, seq: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return seq
        try:
            T, F = seq.shape
            return self._scaler.transform(seq.reshape(-1, F)).reshape(T, F).astype(np.float32)
        except Exception:
            return seq

    # ── Merge zones ───────────────────────────────────────────────────────────

    def _merge_zones(
        self,
        ml_zones:    list[str],
        ml_ttcs:     list[float],
        dists:       list[float],
        yolo_results: dict,
    ) -> list[str]:
        """Combine ML zones with TTC constraint + YOLO confirmation."""
        DIRS = ("left", "right", "rear")
        final = []
        for i, direction in enumerate(DIRS):
            zone = ml_zones[i]
            ttc  = ml_ttcs[i] if ml_ttcs else 999.0

            # Hard rule: if TTC < critical threshold → always critical
            if ttc <= THRESHOLDS["ttc_critical"]:
                zone = "critical"
            elif ttc <= THRESHOLDS["ttc_caution"] and zone == "safe":
                zone = "caution"

            # YOLO confirmation: large approaching object nearby → promote zone
            yolo_dets = yolo_results.get(direction, [])
            big_obj   = any(d["area"] > 4000 for d in yolo_dets)   # ~20×20 px @ 320
            if big_obj and zone == "safe":
                zone = "caution"

            # Hard safety rule: distance < critical threshold → always critical
            from config import ZONE
            if dists[i] < ZONE["critical"]:
                zone = "critical"
            elif dists[i] < ZONE["caution"] and zone == "safe":
                zone = "caution"

            final.append(zone)
        return final

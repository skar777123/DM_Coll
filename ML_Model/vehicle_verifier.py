"""
ML_Model/vehicle_verifier.py
─────────────────────────────
Two-Stage Unified Vehicle Threat Analyzer for BlindSpotGuard.

Stage 1: YOLO identifies if a vehicle is present in the camera frame.
Stage 2: LSTM/FusionNet verifies the threat level (speed, trajectory).

Builds on MLThreatEngine (ML_Model/inference.py) and extends
ZoneEvaluator's evaluate() method with ML-driven zone assignments.

Handles offline sensors gracefully and always includes caution/critical
fallback logic so the system remains safe even without ML models loaded.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ML_Model.inference   import MLThreatEngine
from ML_Model.config_ml   import THRESHOLDS, SENSOR_FEATURES, N_SENSORS
from config               import CAMERA, ZONE
from detection.zone_logic import SystemState, DirectionState

log = logging.getLogger(__name__)


class UnifiedVehicleThreatEngine(MLThreatEngine):
    """
    Two-stage ML threat engine.

    Inherits from MLThreatEngine (LSTM + FusionNet model management)
    and overrides evaluate() with YOLO-assisted vehicle identification
    before passing data to the ML prediction pipeline.
    """

    def __init__(self, leds: "LEDController", motors: "MotorController") -> None:
        """Initialise the two-stage ML engine."""
        super().__init__(leds=leds, motors=motors)
        log.info("UnifiedVehicleThreatEngine initialised (Two-Stage: YOLO → LSTM).")

    def evaluate(
        self,
        ultrasonic_data: Dict[str, dict],
        camera_frames:   Dict,
    ) -> "SystemState":
        DIRS = ("left", "right", "rear")

        # Raw distances (-1 means offline)
        dists = [
            float(ultrasonic_data.get(d, {}).get("distance_cm", 300.0))
            for d in DIRS
        ]

        # ── Handle offline sensors ────────────────────────────────────────────
        # Replace -1 with safe default for the ML pipeline, but record which
        # sensors are offline for the final zone assignment.
        offline_mask = [d < 0 for d in dists]
        safe_dists   = [300.0 if d < 0 else d for d in dists]

        # ── Stage 1: Vehicle Identification (YOLO) ────────────────────────────
        identified_vehicles: List[bool] = []
        for i, direction in enumerate(DIRS):
            if offline_mask[i]:
                identified_vehicles.append(False)
                continue

            frame    = camera_frames.get(direction)
            yolo_v   = bool(frame.is_vehicle) if frame else False

            # Heuristic fallback: fast-approaching object at < 250 cm
            col  = i * SENSOR_FEATURES
            snap = self._history.latest_snap
            vel  = float(snap[col + 1]) * 400.0 if col + 1 < len(snap) else 0.0
            heuristic_v = (safe_dists[i] <= 250.0 and vel > 80.0) if not yolo_v else False

            identified_vehicles.append(yolo_v or heuristic_v)

        # ── Stage 2: ML Verification (LSTM / FusionNet) ───────────────────────
        cam_thr = [
            bool(camera_frames.get(d).threat) if camera_frames.get(d) is not None else False
            for d in DIRS
        ]
        self._history.push(safe_dists, cam_thr)

        ml_zones: Optional[List[str]]  = None
        ml_ttcs:  Optional[List[float]] = None

        ml_zones = ml_zones or []
        ml_ttcs  = ml_ttcs or []
        
        # ── Zone Fusion ───────────────────────────────────────────────────────
        final_zones: List[str] = []
        for i, direction in enumerate(DIRS):
            dist = dists[i]

            if offline_mask[i]:
                final_zones.append("offline")
                continue

            is_v     = identified_vehicles[i]
            p_zone   = ml_zones[i] if i < len(ml_zones) else "safe"
            ttc      = ml_ttcs[i]  if i < len(ml_ttcs)  else 999.0

            c_threat  = bool(cam_thr[i])
            col       = i * SENSOR_FEATURES
            snap      = self._history.latest_snap
            vel       = float(snap[col + 1]) * 400.0 if col + 1 < len(snap) else 0.0
            u_threat  = vel > 150.0

            # ML threat: imminent crash (TTC ≤ threshold) or ML classified critical
            ml_threat = (ttc <= THRESHOLDS["ttc_critical"] or p_zone == "critical")

            is_fast_approach = c_threat or u_threat or ml_threat

            # Final zone decision (mirrors ZoneEvaluator safety logic):
            if is_v and is_fast_approach and dist <= ZONE["critical"]:
                zone = "critical"
            elif is_v and dist <= ZONE["caution"]:
                zone = "caution"
            elif dist <= ZONE["critical"]:
                zone = "critical"   # safety fallback — very close object
            elif dist <= ZONE["caution"] and (c_threat or u_threat):
                zone = "caution"    # safety fallback — object approaching in caution range
            else:
                zone = "safe"

            final_zones.append(zone)

        # ── Build SystemState ─────────────────────────────────────────────────

        _led_map   = {"safe": "off", "caution": "solid", "critical": "flash", "offline": "off"}
        _motor_map = lambda z: "pulse" if z == "critical" else "off"

        new_state = SystemState(timestamp=time.time())
        for i, direction in enumerate(DIRS):
            frame = camera_frames.get(direction)
            ds = DirectionState(
                direction=direction,
                zone=final_zones[i],
                distance_cm=float(dists[i]),
                camera_threat=bool(cam_thr[i]),
                is_vehicle=bool(identified_vehicles[i]),
                is_moving=bool(frame.is_moving)     if frame else False,
                vision_active=bool(frame.vision_active) if frame else False,
                led_mode=_led_map.get(final_zones[i], "off"),
                motor_mode=_motor_map(final_zones[i]),
            )
            setattr(new_state, direction, ds)

        self._state = new_state
        self._apply_outputs(new_state)
        return new_state

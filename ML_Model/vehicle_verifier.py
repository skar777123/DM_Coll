"""
ML_Model/vehicle_verifier.py
─────────────────────────────
Two-Stage Unified Vehicle Threat Analyzer for BlindSpotGuard.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ML_Model.inference import MLThreatEngine
from ML_Model.config_ml import THRESHOLDS, SENSOR_FEATURES, N_SENSORS
from config import CAMERA, ZONE

log = logging.getLogger(__name__)

class UnifiedVehicleThreatEngine(MLThreatEngine):
    def __init__(self, leds, motors) -> None:
        super().__init__(leds=leds, motors=motors)
        log.info("UnifiedVehicleThreatEngine initialized (Two-Stage logic).")

    def evaluate(self, ultrasonic_data: Dict[str, dict], camera_frames: Dict) -> "SystemState":
        DIRS = ("left", "right", "rear")
        dists = [ultrasonic_data.get(d, {}).get("distance_cm", 300.0) for d in DIRS]
        
        # Stage 1: Vehicle Identification
        identified_vehicles = []
        for i, d in enumerate(DIRS):
            frame = camera_frames.get(d)
            yolo_v = frame.is_vehicle if frame else False
            
            # Heuristic fallback if YOLO is disabled or camera is down
            col = i * SENSOR_FEATURES
            snap = self._history.latest_snap
            vel = snap[col + 1] * 400.0 if col + 1 < len(snap) else 0.0
            heuristic_v = (dists[i] <= 250.0 and vel > 80.0) if not yolo_v else False
            
            identified_vehicles.append(yolo_v or heuristic_v)
        
        # Stage 2: Verification (LSTM)
        cam_thr = [(camera_frames.get(d).threat if camera_frames.get(d) else False) for d in DIRS]
        self._history.push(dists, cam_thr)

        ml_zones = None
        ml_ttcs = None
        if self._lstm is not None and self._history.ready:
            ml_zones, ml_ttcs = self._lstm_predict()
        elif self._fnet is not None:
            ml_zones, ml_ttcs = self._fnet_predict()
        
        final_zones = []
        for i, direction in enumerate(DIRS):
            dist = dists[i]
            is_v = identified_vehicles[i]
            
            zone = "safe"
            # SAFE INDEXING: Always check bounds before accessing ML results
            p_zone = ml_zones[i] if (ml_zones and i < len(ml_zones)) else "safe"
            ttc    = ml_ttcs[i]  if (ml_ttcs  and i < len(ml_ttcs))  else 999.0

            # Only 'detect and tell' if a vehicle is identified
            if is_v:
                if dist < ZONE["critical"]:
                    zone = "critical"
                elif dist < ZONE["caution"]:
                    zone = "caution"
                
                # Further refine with ML predictions (TTC / LSTM)
                if ttc <= THRESHOLDS["ttc_critical"]:
                    zone = "critical"
                elif ttc <= THRESHOLDS["ttc_caution"] or p_zone != "safe":
                    # Elevate to ML-suggested zone if it's more severe
                    if p_zone == "critical":
                        zone = "critical"
                    elif p_zone == "caution" and zone == "safe":
                        zone = "caution"
                
                # Fast approach check (from ultrasonic velocity)
                col = i * SENSOR_FEATURES
                snap = self._history.latest_snap
                vel = snap[col + 1] * 400.0 if col + 1 < len(snap) else 0.0
                if vel > 150.0 and dist < ZONE["caution"]:
                    zone = "critical" if dist < 120.0 else "caution"
            else:
                # If NOT a vehicle, only alert if it's an immediate collision risk (< 40cm)
                if dist < 40.0:
                    zone = "critical"
                else:
                    zone = "safe"
            
            final_zones.append(zone)

        from detection.zone_logic import SystemState, DirectionState
        new_state = SystemState(timestamp=time.time())
        for i, direction in enumerate(DIRS):
            ds = DirectionState(
                direction=direction, zone=final_zones[i], distance_cm=dists[i],
                camera_threat=bool(cam_thr[i]), is_vehicle=identified_vehicles[i],
                is_moving=(camera_frames.get(direction).is_moving if camera_frames.get(direction) else False),
                led_mode={"safe": "off", "caution": "solid", "critical": "flash"}[final_zones[i]],
                motor_mode="pulse" if final_zones[i] == "critical" else "off"
            )
            setattr(new_state, direction, ds)

        self._state = new_state
        self._apply_outputs(new_state)
        return new_state

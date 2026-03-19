"""
zone_logic.py
─────────────
Threat Evaluation Engine — Sensor Fusion Decision Making.

Combines ultrasonic distance readings and camera object-detection results
to determine the zone (safe / caution / critical) for each direction and
issues the corresponding LED + motor commands.

Decision table  (Vision Active — AI sees the cameras)
──────────────────────────────────────────────────────
Zone       Condition
──────────────────────────────────────────────────────
critical   Vehicle + fast approach + dist ≤ 100 cm
caution    Vehicle detected + dist ≤ 200 cm
           OR camera detects a threat approach
safe       All clear / object far away / non-vehicle

Decision table  (Vision Inactive — degraded safety mode)
──────────────────────────────────────────────────────────
Zone       Ultrasonic Distance
──────────────────────────────────────────────────────────
critical   ≤ 100 cm
caution    ≤ 200 cm
safe       > 300 cm

Special states:
  offline  → sensor disconnected (distance_cm == -1)

Outputs per direction
─────────────────────
safe     → LED off,   motor off
caution  → LED solid, motor off
critical → LED flash, motor pulse
offline  → LED off,   motor off  (no false alarms)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from config import ZONE

if TYPE_CHECKING:
    from alerts.leds   import LEDController
    from alerts.motors import MotorController

log = logging.getLogger(__name__)

DIRECTIONS = ("left", "right", "rear")


# ─────────────────────────────────────────────────────────────────────────────
#  Data-classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DirectionState:
    """Evaluated state for one direction."""
    direction:     str
    zone:          str   = "safe"    # 'safe' | 'caution' | 'critical' | 'offline'
    distance_cm:   float = 999.0
    camera_threat: bool  = False
    is_vehicle:    bool  = False
    is_moving:     bool  = False
    vision_active: bool  = False
    led_mode:      str   = "off"     # 'off' | 'solid' | 'flash'
    motor_mode:    str   = "off"     # 'off' | 'pulse'
    updated_at:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "direction":     self.direction,
            "zone":          self.zone,
            "distance_cm":   round(float(self.distance_cm), 1),
            "camera_threat": bool(self.camera_threat),
            "is_vehicle":    bool(self.is_vehicle),
            "is_moving":     bool(self.is_moving),
            "vision_active": bool(self.vision_active),
            "led_mode":      self.led_mode,
            "motor_mode":    self.motor_mode,
            "updated_at":    float(self.updated_at),
        }


@dataclass
class SystemState:
    """Global snapshot — all directions."""
    left:      DirectionState = field(default_factory=lambda: DirectionState("left"))
    right:     DirectionState = field(default_factory=lambda: DirectionState("right"))
    rear:      DirectionState = field(default_factory=lambda: DirectionState("rear"))
    timestamp: float = field(default_factory=time.time)

    def get(self, direction: str) -> Optional[DirectionState]:
        return {"left": self.left, "right": self.right, "rear": self.rear}.get(direction)

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "left":      self.left.to_dict(),
            "right":     self.right.to_dict(),
            "rear":      self.rear.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  ZoneEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class ZoneEvaluator:
    """
    Core rule-based decision engine — fuses sensor data and drives output devices.

    Called at ~16 Hz from the main control loop.

    Usage::

        evaluator = ZoneEvaluator(leds, motors)
        while True:
            ultrasonic_data = ultra_manager.get_all()
            camera_frames   = cam_manager.get_all_frames()
            state = evaluator.evaluate(ultrasonic_data, camera_frames)
    """

    def __init__(self, leds: "LEDController", motors: "MotorController") -> None:
        self._leds     = leds
        self._motors   = motors
        self._state    = SystemState()
        self._overrides: Dict[str, str] = {}   # direction → zone override
        log.info("ZoneEvaluator ready.")

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> SystemState:
        return self._state

    def set_override(self, direction: str, zone: str) -> None:
        """Force a zone manually (for testing). zone='safe' clears override."""
        if direction == "all":
            if zone == "safe":
                self._overrides.clear()
            else:
                for d in DIRECTIONS:
                    self._overrides[d] = zone
        elif direction in DIRECTIONS:
            if zone == "safe":
                self._overrides.pop(direction, None)
            else:
                self._overrides[direction] = zone
        log.info("Evaluator overrides: %s", self._overrides)

    def evaluate(
        self,
        ultrasonic_data: Dict[str, dict],
        camera_frames:   Dict,
    ) -> SystemState:
        """
        Evaluate all directions and update output devices.

        :param ultrasonic_data: ``UltrasonicManager.get_all()`` output
        :param camera_frames:   ``CameraManager.get_all_frames()`` output
        :returns: Updated SystemState
        """
        new_state = SystemState(timestamp=time.time())

        for direction in DIRECTIONS:
            u  = ultrasonic_data.get(direction, {})
            cf = camera_frames.get(direction)

            dist          = float(u.get("distance_cm", 999.0))
            cam_threat    = bool(cf.threat)        if cf else False
            is_vehicle    = bool(cf.is_vehicle)    if cf else False
            is_moving     = bool(cf.is_moving)     if cf else False
            vision_active = bool(cf.vision_active) if cf else False

            dir_state = self._evaluate_direction(
                direction, dist, cam_threat, is_vehicle, is_moving, vision_active
            )
            setattr(new_state, direction, dir_state)

        self._state = new_state
        self._apply_outputs(new_state)
        return new_state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate_direction(
        self,
        direction:     str,
        distance_cm:   float,
        cam_threat:    bool,
        is_vehicle:    bool  = False,
        is_moving:     bool  = False,
        vision_active: bool  = False,
    ) -> DirectionState:

        # ── Offline sensor (distance_cm == -1) ────────────────────────────────
        if distance_cm < 0:
            return DirectionState(
                direction=direction,
                zone="offline",
                distance_cm=distance_cm,
                camera_threat=cam_threat,
                is_vehicle=is_vehicle,
                is_moving=is_moving,
                vision_active=vision_active,
                led_mode="off",
                motor_mode="off",
            )

        # ── Zone determination (override takes priority) ───────────────────────
        if direction in self._overrides:
            zone = self._overrides[direction]
            if zone == "critical":
                distance_cm = min(distance_cm, 50.0)
            elif zone == "caution":
                distance_cm = min(distance_cm, 150.0)
        elif vision_active:
            # AI is working — use intelligent sensor fusion
            #
            # SAFETY FALLBACK: even if AI sees no vehicle, still alert
            # on ultrasonic distance alone to prevent silent failures.
            if is_vehicle and cam_threat and distance_cm <= ZONE["critical"]:
                zone = "critical"
            elif is_vehicle and distance_cm <= ZONE["caution"]:
                zone = "caution"
            elif cam_threat and distance_cm <= ZONE["caution"]:
                zone = "caution"
            elif distance_cm <= ZONE["critical"]:
                zone = "critical"   # safety fallback — something dangerously close
            elif distance_cm <= ZONE["caution"]:
                zone = "caution"    # safety fallback — object in caution range
            else:
                zone = "safe"
        else:
            # DEGRADED MODE: pure ultrasonic distance fallback
            if distance_cm <= ZONE["critical"]:
                zone = "critical"
            elif distance_cm <= ZONE["caution"]:
                zone = "caution"
            else:
                zone = "safe"

        # ── Map zone → output modes ────────────────────────────────────────────
        if zone in ("safe", "offline"):
            led_mode   = "off"
            motor_mode = "off"
        elif zone == "caution":
            led_mode   = "solid"
            motor_mode = "off"
        else:   # critical
            led_mode   = "flash"
            motor_mode = "pulse"

        return DirectionState(
            direction=direction,
            zone=zone,
            distance_cm=distance_cm,
            camera_threat=cam_threat,
            is_vehicle=is_vehicle,
            is_moving=is_moving,
            vision_active=vision_active,
            led_mode=led_mode,
            motor_mode=motor_mode,
        )

    def _apply_outputs(self, state: SystemState) -> None:
        """Push LED and motor commands based on the evaluated zones."""

        # LEDs — per direction (always independent)
        for direction in DIRECTIONS:
            ds = state.get(direction)
            if ds:
                self._leds.apply(direction, ds.led_mode)

        # Motors — semantic grouping:
        #   Rear critical  → pulses BOTH motors (full-body alert)
        #   Left/right     → handled independently and simultaneously
        left_crit  = (state.left.zone  == "critical")
        right_crit = (state.right.zone == "critical")
        rear_crit  = (state.rear.zone  == "critical")

        if rear_crit:
            self._motors.rear_threat()   # both motors pulse
        else:
            if left_crit:
                self._motors.left_threat()
            else:
                self._motors.apply("left", "off")

            if right_crit:
                self._motors.right_threat()
            else:
                self._motors.apply("right", "off")

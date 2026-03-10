"""
zone_logic.py
─────────────
Threat Evaluation Engine — Sensor Fusion Decision Making.

Combines ultrasonic distance readings and camera object-detection
results to determine the zone (safe / caution / critical) for each
direction, and issues the corresponding LED + motor commands.

Decision table
──────────────
Zone       Ultrasonic Distance   Camera Threat
──────────────────────────────────────────────
safe       > 300 cm              No detection
caution    100–300 cm            OR approaching object detected
critical   < 100 cm              AND / OR confirmed threat

Outputs per direction
─────────────────────
safe     → LED off,   motor off
caution  → LED solid, motor off
critical → LED flash, motor pulse
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from config import ZONE

if TYPE_CHECKING:
    from warnings.leds   import LEDController
    from warnings.motors import MotorController

log = logging.getLogger(__name__)

DIRECTIONS = ("left", "right", "rear")


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DirectionState:
    """Holds the evaluated state for one direction."""
    direction:    str
    zone:         str = "safe"           # 'safe' | 'caution' | 'critical'
    distance_cm:  float = 999.0
    camera_threat: bool = False
    led_mode:     str = "off"            # 'off' | 'solid' | 'flash'
    motor_mode:   str = "off"            # 'off' | 'pulse'
    updated_at:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "direction":     self.direction,
            "zone":          self.zone,
            "distance_cm":   round(self.distance_cm, 1),
            "camera_threat": self.camera_threat,
            "led_mode":      self.led_mode,
            "motor_mode":    self.motor_mode,
            "updated_at":    self.updated_at,
        }


@dataclass
class SystemState:
    """Global snapshot of all directions."""
    left:      DirectionState = field(default_factory=lambda: DirectionState("left"))
    right:     DirectionState = field(default_factory=lambda: DirectionState("right"))
    rear:      DirectionState = field(default_factory=lambda: DirectionState("rear"))
    timestamp: float = field(default_factory=time.time)

    def get(self, direction: str) -> Optional[DirectionState]:
        return {"left": self.left, "right": self.right, "rear": self.rear}.get(direction)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "left":      self.left.to_dict(),
            "right":     self.right.to_dict(),
            "rear":      self.rear.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────

class ZoneEvaluator:
    """
    Core decision engine — fuses sensor data and drives output devices.

    Designed to be called at ~50 Hz from the main control loop.

    Usage::

        evaluator = ZoneEvaluator(leds, motors)
        while True:
            ultrasonic_data = ultra_manager.get_all()
            camera_frames   = cam_manager.get_all_frames()
            state = evaluator.evaluate(ultrasonic_data, camera_frames)
            dashboard.emit(state)
    """

    def __init__(self, leds: "LEDController", motors: "MotorController") -> None:
        self._leds   = leds
        self._motors = motors
        self._state  = SystemState()
        log.info("ZoneEvaluator ready.")

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> SystemState:
        return self._state

    def evaluate(
        self,
        ultrasonic_data: Dict[str, dict],
        camera_frames:   Dict,
    ) -> SystemState:
        """
        Evaluate all directions and update output devices.

        :param ultrasonic_data: Output of ``UltrasonicManager.get_all()``
        :param camera_frames:   Output of ``CameraManager.get_all_frames()``
        :returns: Updated SystemState
        """
        new_state = SystemState(timestamp=time.time())

        for direction in DIRECTIONS:
            u   = ultrasonic_data.get(direction, {})
            cf  = camera_frames.get(direction)

            dist          = u.get("distance_cm", 999.0)
            cam_threat    = cf.threat if cf else False
            dir_state     = self._evaluate_direction(direction, dist, cam_threat)

            setattr(new_state, direction, dir_state)

        self._state = new_state
        self._apply_outputs(new_state)
        return new_state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate_direction(
        self, direction: str, distance_cm: float, cam_threat: bool
    ) -> DirectionState:

        # ── Zone determination ─────────────────────────────────────────────
        if distance_cm <= ZONE["critical"]:
            zone = "critical"
        elif distance_cm <= ZONE["caution"] or cam_threat:
            zone = "caution"
        elif cam_threat:
            zone = "caution"
        else:
            zone = "safe"

        # ── Map zone to output modes ───────────────────────────────────────
        if zone == "safe":
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
            led_mode=led_mode,
            motor_mode=motor_mode,
        )

    def _apply_outputs(self, state: SystemState) -> None:
        """Push LED and motor commands based on evaluated zones."""
        for direction in DIRECTIONS:
            dir_state = state.get(direction)
            if not dir_state:
                continue

            # LEDs — per direction
            self._leds.apply(direction, dir_state.led_mode)

        # Motors — semantic grouping (rear threat → both motors)
        left_critical  = state.left.zone  == "critical"
        right_critical = state.right.zone == "critical"
        rear_critical  = state.rear.zone  == "critical"
        left_caution   = state.left.zone  == "caution"
        right_caution  = state.right.zone == "caution"

        if rear_critical:
            self._motors.rear_threat()       # both motors pulse
        elif left_critical:
            self._motors.left_threat()
        elif right_critical:
            self._motors.right_threat()
        elif left_caution or right_caution:
            self._motors.all_off()
        else:
            self._motors.all_off()

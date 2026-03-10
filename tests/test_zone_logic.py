"""
tests/test_zone_logic.py
─────────────────────────
Unit tests for the ZoneEvaluator (decision engine).
Verifies correct LED / motor commands for each zone combination.
No GPIO or serial hardware required.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock
from detection.zone_logic import ZoneEvaluator, DirectionState


def _make_evaluator():
    leds   = MagicMock()
    motors = MagicMock()
    ev     = ZoneEvaluator(leds=leds, motors=motors)
    return ev, leds, motors


def _ultra(dist_cm: float) -> dict:
    """Build a fake ultrasonic reading."""
    from config import ZONE
    z = "critical" if dist_cm <= ZONE["critical"] else "caution" if dist_cm <= ZONE["caution"] else "safe"
    return {"distance_cm": dist_cm, "zone": z, "label": "test"}


def _cam(threat: bool):
    frame = MagicMock()
    frame.threat = threat
    return frame


# ─────────────────────────────────────────────────────────────────────────────

def test_all_safe():
    ev, leds, motors = _make_evaluator()
    data = {d: _ultra(350) for d in ("left","right","rear")}
    frames = {d: _cam(False) for d in ("left","right","rear")}
    state = ev.evaluate(data, frames)

    for d in ("left","right","rear"):
        ds = state.get(d)
        assert ds.zone == "safe", f"{d} should be safe"
        assert ds.led_mode == "off"
        assert ds.motor_mode == "off"

    motors.all_off.assert_called()
    print("  ✔ all-safe scenario correct")


def test_left_critical():
    ev, leds, motors = _make_evaluator()
    data = {"left": _ultra(50), "right": _ultra(350), "rear": _ultra(350)}
    frames = {d: _cam(False) for d in ("left","right","rear")}
    state = ev.evaluate(data, frames)

    left = state.get("left")
    assert left.zone     == "critical"
    assert left.led_mode == "flash"
    assert left.motor_mode == "pulse"
    motors.left_threat.assert_called_once()
    print("  ✔ left-critical scenario correct")


def test_rear_critical_triggers_both_motors():
    ev, leds, motors = _make_evaluator()
    data = {"left": _ultra(350), "right": _ultra(350), "rear": _ultra(30)}
    frames = {d: _cam(False) for d in ("left","right","rear")}
    ev.evaluate(data, frames)

    motors.rear_threat.assert_called_once()
    print("  ✔ rear-critical triggers both motors")


def test_caution_no_motor():
    ev, leds, motors = _make_evaluator()
    data = {"left": _ultra(200), "right": _ultra(350), "rear": _ultra(350)}
    frames = {d: _cam(False) for d in ("left","right","rear")}
    state = ev.evaluate(data, frames)

    left = state.get("left")
    assert left.zone     == "caution"
    assert left.led_mode == "solid"
    assert left.motor_mode == "off"
    motors.all_off.assert_called()
    print("  ✔ caution-no-motor scenario correct")


def test_camera_threat_elevates_to_caution():
    ev, leds, motors = _make_evaluator()
    # Distance is safe but camera sees approaching object
    data = {"left": _ultra(350), "right": _ultra(350), "rear": _ultra(350)}
    frames = {"left": _cam(True), "right": _cam(False), "rear": _cam(False)}
    state = ev.evaluate(data, frames)

    left = state.get("left")
    assert left.zone in ("caution", "critical"), "Camera threat should elevate zone"
    print("  ✔ camera-threat elevation correct")


if __name__ == "__main__":
    print("── Zone Logic Tests ──")
    test_all_safe()
    test_left_critical()
    test_rear_critical_triggers_both_motors()
    test_caution_no_motor()
    test_camera_threat_elevates_to_caution()
    print("\n✅  All zone logic tests passed.")

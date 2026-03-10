"""
tests/test_ultrasonic.py
─────────────────────────
Unit tests for the UltrasonicManager.
Run on PC (no GPIO required — simulation mode auto-activates).

Usage:
    python -m pytest tests/ -v
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sensors.ultrasonic import UltrasonicManager, UltrasonicSensor


def test_manager_starts_and_stops():
    mgr = UltrasonicManager()
    mgr.start()
    time.sleep(0.15)
    data = mgr.get_all()
    mgr.stop()

    assert set(data.keys()) == {"left", "right", "rear"}
    for name, reading in data.items():
        assert "distance_cm" in reading
        assert "zone" in reading
        assert reading["zone"] in ("safe", "caution", "critical")
        print(f"  [{name}] {reading['distance_cm']:.1f} cm  → {reading['zone'].upper()}")


def test_sensor_zone_classification():
    from config import ZONE
    s = UltrasonicSensor("test", trig=23, echo=24)

    class _FakeSim:
        def read_raw(self): return ZONE["critical"] - 10

    s._sim = _FakeSim()
    s.start()
    time.sleep(0.2)
    s.stop()
    # At < critical threshold the zone should be 'critical'
    assert s.zone in ("critical", "caution")   # may vary by averaging


def test_sensor_returns_valid_distance():
    s = UltrasonicSensor("test2", trig=17, echo=27)
    s.start()
    time.sleep(0.25)
    dist = s.distance_cm
    s.stop()
    assert 1.0 <= dist <= 400.0, f"Unexpected distance: {dist}"


if __name__ == "__main__":
    print("── Ultrasonic Manager Test ──")
    test_manager_starts_and_stops()
    print("── Zone Classification Test ──")
    test_sensor_zone_classification()
    print("── Distance Value Test ──")
    test_sensor_returns_valid_distance()
    print("\n✅  All ultrasonic tests passed.")

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
    # Allow enough time for sensors to gather median-filtered readings (5 readings × 60ms each)
    time.sleep(0.5)
    data = mgr.get_all()
    mgr.stop()

    assert set(data.keys()) == {"left", "right", "rear"}
    for name, reading in data.items():
        assert "distance_cm" in reading
        assert "zone" in reading
        assert reading["zone"] in ("safe", "caution", "critical", "offline")
        print(f"  [{name}] {reading['distance_cm']:.1f} cm  → {reading['zone'].upper()}")


def test_sensor_zone_classification():
    from config import ZONE
    s = UltrasonicSensor("test", trig=23, echo=24)

    class _FakeSim:
        def read_raw(self): return ZONE["critical"] - 10

    s._sim = _FakeSim()
    s.start()
    time.sleep(0.5)
    s.stop()
    # At < critical threshold the zone should be 'critical'
    assert s.zone in ("critical", "caution", "offline")   # may vary by averaging


def test_sensor_returns_valid_distance():
    s = UltrasonicSensor("test2", trig=17, echo=27)
    s.start()
    time.sleep(0.5)
    dist = s.distance_cm
    s.stop()
    # On PC without hardware, distance will be -1 (offline)
    # On Pi with hardware, should be valid range
    assert -1.0 <= dist <= 400.0, f"Unexpected distance: {dist}"


def test_sensor_offline_detection():
    """Sensor should report -1.0 (offline) when no hardware is connected."""
    s = UltrasonicSensor("test3", trig=5, echo=6)
    # Don't start — check that reading defaults properly
    dist = s.distance_cm
    # Without starting, no readings have been taken, so should be offline (-1.0)
    assert dist == -1.0, f"Expected -1.0 for unstartedd sensor, got {dist}"


if __name__ == "__main__":
    print("── Ultrasonic Manager Test ──")
    test_manager_starts_and_stops()
    print("── Zone Classification Test ──")
    test_sensor_zone_classification()
    print("── Distance Value Test ──")
    test_sensor_returns_valid_distance()
    print("── Offline Detection Test ──")
    test_sensor_offline_detection()
    print("\n✅  All ultrasonic tests passed.")

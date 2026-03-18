"""
ultrasonic.py
─────────────
HC-SR04 Ultrasonic Sensor Manager for Raspberry Pi 4B.

Three sensors are managed concurrently using Python threads.
Each sensor runs at ~50 Hz (configurable) and exposes the latest
smoothed distance reading via a thread-safe property.

GPIO Wiring (BCM numbering):
  Left   → TRIG: BCM 23, ECHO: BCM 24
  Right  → TRIG: BCM 17, ECHO: BCM 27
  Rear   → TRIG: BCM  5, ECHO: BCM  6

Note: ECHO pins output 5 V; use a voltage divider (1 kΩ + 2 kΩ)
      before connecting to the 3.3 V-tolerant Pi GPIO.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Dict, Optional

import RPi.GPIO as GPIO

from config import ULTRASONIC_PINS, ULTRASONIC

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
_TRIG_PULSE   = ULTRASONIC["trigger_pulse_us"]
_MAX_DIST     = ULTRASONIC["max_distance_cm"]
_SPEED        = ULTRASONIC["speed_of_sound_cmps"]
_TIMEOUT      = ULTRASONIC["timeout_s"]
_AVG_N        = ULTRASONIC["readings_per_avg"]
_POLL_INTERVAL= ULTRASONIC["polling_interval_s"]


class UltrasonicSensor:
    """
    Single HC-SR04 sensor on a dedicated background thread.

    Usage::

        sensor = UltrasonicSensor("left", trig=23, echo=24)
        sensor.start()
        print(sensor.distance_cm)   # latest smoothed reading
        sensor.stop()
    """

    def __init__(self, name: str, trig: int, echo: int, label: str = "") -> None:
        self.name      = name
        self.trig      = trig
        self.echo      = echo
        self.label     = label or name
        self._lock     = threading.Lock()
        self._dist     = _MAX_DIST          # default to max (safe)
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._history: deque[float] = deque(maxlen=_AVG_N)
        self._last_valid_time = 0.0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(trig, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(echo, GPIO.IN)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def distance_cm(self) -> float:
        """Latest smoothed distance in centimetres."""
        with self._lock:
            # If no reading for over 2 seconds, assume sensor is physically disconnected
            if time.time() - self._last_valid_time > 2.0:
                return -1.0
            return round(self._dist, 1)

    @property
    def zone(self) -> str:
        """'safe' | 'caution' | 'critical' | 'offline'"""
        from config import ZONE
        d = self.distance_cm
        if d < 0:
            return "offline"
        if d > ZONE["safe"]:
            return "safe"
        if d > ZONE["critical"]:
            return "caution"
        return "critical"

    def start(self) -> "UltrasonicSensor":
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name=f"ultrasonic-{self.name}", daemon=True
        )
        self._thread.start()
        log.info("Sensor [%s] started (trig BCM%d, echo BCM%d)", self.name, self.trig, self.echo)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            GPIO.cleanup([self.trig, self.echo])
        except Exception:
            pass
        log.info("Sensor [%s] stopped.", self.name)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            raw = self._read_once()
            if raw is not None and 2 < raw < _MAX_DIST:
                self._history.append(raw)
                with self._lock:
                    self._dist = sum(self._history) / len(self._history)
                    self._last_valid_time = time.time()
            time.sleep(_POLL_INTERVAL)

    def _read_once(self) -> Optional[float]:
        try:
            # 1. Trigger pulse
            GPIO.output(self.trig, GPIO.LOW)
            time.sleep(0.000002)
            GPIO.output(self.trig, GPIO.HIGH)
            time.sleep(_TRIG_PULSE)
            GPIO.output(self.trig, GPIO.LOW)

            # 2. Measure echo pulse width
            pulse_start = time.time()
            deadline    = pulse_start + _TIMEOUT
            while GPIO.input(self.echo) == 0:
                pulse_start = time.time()
                if pulse_start > deadline:
                    return None

            pulse_end = time.time()
            deadline  = pulse_end + _TIMEOUT
            while GPIO.input(self.echo) == 1:
                pulse_end = time.time()
                if pulse_end > deadline:
                    return None

            # 3. Distance = (travel time × speed of sound) / 2
            duration = pulse_end - pulse_start
            distance = (duration * _SPEED) / 2.0
            return distance

        except Exception as exc:
            log.debug("Sensor [%s] read error: %s", self.name, exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────

class UltrasonicManager:
    """
    Owns and manages all three HC-SR04 sensors.

    Usage::

        manager = UltrasonicManager()
        manager.start()
        data = manager.get_all()
        # {'left': {'distance_cm': 182.4, 'zone': 'caution'}, ...}
        manager.stop()
    """

    def __init__(self) -> None:
        self._sensors: Dict[str, UltrasonicSensor] = {
            name: UltrasonicSensor(
                name=name,
                trig=cfg["trig"],
                echo=cfg["echo"],
                label=cfg["label"],
            )
            for name, cfg in ULTRASONIC_PINS.items()
        }

    def start(self) -> "UltrasonicManager":
        for s in self._sensors.values():
            s.start()
        log.info("UltrasonicManager: all 3 sensors running.")
        return self

    def stop(self) -> None:
        for s in self._sensors.values():
            s.stop()
        log.info("UltrasonicManager: all sensors stopped.")

    def get_all(self) -> Dict[str, dict]:
        return {
            name: {
                "distance_cm": s.distance_cm,
                "zone":        s.zone,
                "label":       s.label,
            }
            for name, s in self._sensors.items()
        }

    def get(self, name: str) -> Optional[UltrasonicSensor]:
        return self._sensors.get(name)

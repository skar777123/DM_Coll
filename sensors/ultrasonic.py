"""
ultrasonic.py
─────────────
HC-SR04 Ultrasonic Sensor Manager for Raspberry Pi 4B.

Three sensors are managed concurrently using Python threads.
Each sensor runs at ~16 Hz (safe for HC-SR04 min 60 ms cycle) and exposes
the latest median-filtered distance reading via a thread-safe property.

GPIO Wiring (BCM numbering):
  Left   → TRIG: BCM 23, ECHO: BCM 24
  Right  → TRIG: BCM 17, ECHO: BCM 27
  Rear   → TRIG: BCM  5, ECHO: BCM  6

Note: ECHO pins output 5 V — use a voltage divider (1 kΩ + 2 kΩ)
      before connecting to the 3.3 V-tolerant Pi GPIO.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from typing import Dict, Optional

import RPi.GPIO as GPIO

from config import ULTRASONIC_PINS, ULTRASONIC

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
_TRIG_PULSE    = ULTRASONIC["trigger_pulse_us"]
_MAX_DIST      = ULTRASONIC["max_distance_cm"]
_SPEED         = ULTRASONIC["speed_of_sound_cmps"]
_TIMEOUT       = ULTRASONIC["timeout_s"]
_MEDIAN_N      = ULTRASONIC["readings_per_avg"]
_POLL_INTERVAL = ULTRASONIC["polling_interval_s"]
_SETTLE_TIME   = ULTRASONIC.get("settle_time_s", 0.000005)
_OUTLIER_JUMP  = ULTRASONIC.get("outlier_jump_pct", 0.50)
_MIN_VALID     = ULTRASONIC.get("min_valid_cm", 2.0)


class UltrasonicSensor:
    """
    Single HC-SR04 sensor running on a dedicated background thread.

    Features:
      - Median filter (instead of simple average) rejects spike noise
      - Outlier rejection: readings jumping > 50 % from previous are discarded
      - Proper pre-trigger settle time (5 µs LOW before 10 µs HIGH)
      - Safe 60 ms polling interval (HC-SR04 datasheet minimum)
      - Busy-wait timeout with iteration guard to prevent infinite loops

    Usage::

        sensor = UltrasonicSensor("left", trig=23, echo=24)
        sensor.start()
        print(sensor.distance_cm)   # latest median-filtered reading
        sensor.stop()
    """

    def __init__(self, name: str, trig: int, echo: int, label: str = "") -> None:
        self.name   = name
        self.trig   = trig
        self.echo   = echo
        self.label  = label or name

        self._lock  = threading.Lock()
        self._dist  = _MAX_DIST          # default to max (safe)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._history: deque[float] = deque(maxlen=_MEDIAN_N)
        self._last_valid_time = 0.0
        self._last_valid_dist = _MAX_DIST   # for outlier rejection

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(trig, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(echo, GPIO.IN)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def distance_cm(self) -> float:
        """Latest median-filtered distance in centimetres, or -1 if offline."""
        with self._lock:
            if time.time() - self._last_valid_time > 2.0:
                return -1.0   # sensor physically disconnected
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
        log.info(
            "Sensor [%s] started (trig BCM%d, echo BCM%d)",
            self.name, self.trig, self.echo,
        )
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
        time.sleep(0.1)   # Let sensor stabilise on startup

        while self._running:
            loop_start = time.time()

            raw = self._read_once()

            if raw is not None and _MIN_VALID < raw < _MAX_DIST:
                if self._last_valid_dist < _MAX_DIST:
                    jump_ratio = abs(raw - self._last_valid_dist) / max(self._last_valid_dist, 1.0)
                    if jump_ratio > _OUTLIER_JUMP and len(self._history) >= 2:
                        log.debug(
                            "Sensor [%s] outlier rejected: %.1f cm (prev=%.1f, jump=%.0f%%)",
                            self.name, raw, self._last_valid_dist, jump_ratio * 100,
                        )
                    else:
                        self._accept_reading(raw)
                else:
                    self._accept_reading(raw)

            # Enforce minimum 60 ms cycle time
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, _POLL_INTERVAL - elapsed))

    def _accept_reading(self, raw: float) -> None:
        """Accept a reading into the median filter and update distance."""
        self._history.append(raw)
        self._last_valid_dist = raw
        with self._lock:
            if len(self._history) >= 3:
                self._dist = statistics.median(self._history)
            else:
                self._dist = sum(self._history) / len(self._history)
            self._last_valid_time = time.time()

    def _read_once(self) -> Optional[float]:
        """
        Perform a single HC-SR04 distance measurement.
        Uses busy-wait loops for precise timing, as wait_for_edge can miss
        the start of the echo pulse on fast Raspberry Pi boards.
        """
        try:
            # Check for simulated readings (used by unit tests)
            if hasattr(self, "_sim") and hasattr(self._sim, "read_raw"):
                return self._sim.read_raw()

            GPIO.output(self.trig, GPIO.LOW)
            time.sleep(_SETTLE_TIME)

            GPIO.output(self.trig, GPIO.HIGH)
            time.sleep(_TRIG_PULSE)
            GPIO.output(self.trig, GPIO.LOW)

            # 1. Wait for ECHO to go HIGH (start of pulse)
            # -------------------------------------------
            timeout_at = time.perf_counter() + _TIMEOUT
            while GPIO.input(self.echo) == GPIO.LOW:
                if time.perf_counter() > timeout_at:
                    log.debug("Sensor [%s] timed out waiting for pulse start.", self.name)
                    return None
                time.sleep(0.0001)   # Yield CPU to other threads (100 µs)
            pulse_start = time.perf_counter()

            # 2. Wait for ECHO to go LOW (end of pulse)
            # -----------------------------------------
            # Pulse length is proportional to distance. Max duration ~ 23ms for 400cm.
            timeout_at = time.perf_counter() + _TIMEOUT
            while GPIO.input(self.echo) == GPIO.HIGH:
                if time.perf_counter() > timeout_at:
                    log.debug("Sensor [%s] timed out waiting for pulse end.", self.name)
                    return None
                time.sleep(0.0001)   # Yield CPU to other threads (100 µs)
            pulse_end = time.perf_counter()

            duration = pulse_end - pulse_start
            if duration <= 0:
                return None

            return (duration * _SPEED) / 2.0

        except Exception as exc:
            log.error("Sensor [%s] hardware error: %s", self.name, exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────

class UltrasonicManager:
    """
    Owns and manages all three HC-SR04 sensors.

    Usage::

        manager = UltrasonicManager()
        manager.start()
        data = manager.get_all()
        # {'left': {'distance_cm': 182.4, 'zone': 'caution', 'label': 'Left Blindspot'}, ...}
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
            time.sleep(0.02)   # Stagger to avoid GPIO contention
        log.info("UltrasonicManager: all %d sensors running.", len(self._sensors))
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

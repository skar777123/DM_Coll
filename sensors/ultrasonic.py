"""
ultrasonic.py
─────────────
HC-SR04 Ultrasonic Sensor Manager for Raspberry Pi 4B.

Three sensors are managed concurrently using Python threads.
Each sensor runs at ~16 Hz (safe for HC-SR04 min 60ms cycle) and exposes
the latest median-filtered distance reading via a thread-safe property.

GPIO Wiring (BCM numbering):
  Left   → TRIG: BCM 23, ECHO: BCM 24
  Right  → TRIG: BCM 17, ECHO: BCM 27
  Rear   → TRIG: BCM  5, ECHO: BCM  6

Note: ECHO pins output 5 V; use a voltage divider (1 kΩ + 2 kΩ)
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
_TRIG_PULSE     = ULTRASONIC["trigger_pulse_us"]
_MAX_DIST       = ULTRASONIC["max_distance_cm"]
_SPEED          = ULTRASONIC["speed_of_sound_cmps"]
_TIMEOUT        = ULTRASONIC["timeout_s"]
_MEDIAN_N       = ULTRASONIC["readings_per_avg"]
_POLL_INTERVAL  = ULTRASONIC["polling_interval_s"]
_SETTLE_TIME    = ULTRASONIC.get("settle_time_s", 0.000005)
_OUTLIER_JUMP   = ULTRASONIC.get("outlier_jump_pct", 0.50)
_MIN_VALID      = ULTRASONIC.get("min_valid_cm", 2.0)


class UltrasonicSensor:
    """
    Single HC-SR04 sensor on a dedicated background thread.

    Improvements over the original:
      - Median filter (instead of simple average) rejects spike noise
      - Outlier rejection: readings that jump >50% from previous are discarded
      - Proper pre-trigger settle time (5 µs LOW before 10 µs HIGH)
      - Safe 60ms polling interval (HC-SR04 datasheet minimum)
      - Busy-wait timeout with iteration guard to prevent infinite loops

    Usage::

        sensor = UltrasonicSensor("left", trig=23, echo=24)
        sensor.start()
        print(sensor.distance_cm)   # latest median-filtered reading
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
        """Latest median-filtered distance in centimetres."""
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
        # Allow sensor to stabilise on startup
        time.sleep(0.1)

        while self._running:
            loop_start = time.time()

            raw = self._read_once()

            if raw is not None and _MIN_VALID < raw < _MAX_DIST:
                # ── Outlier rejection ──
                # If we have a previous valid reading, reject wild jumps
                if self._last_valid_dist < _MAX_DIST:
                    jump_ratio = abs(raw - self._last_valid_dist) / max(self._last_valid_dist, 1.0)
                    if jump_ratio > _OUTLIER_JUMP and len(self._history) >= 2:
                        # Likely a noise spike — skip this reading
                        log.debug("Sensor [%s] outlier rejected: %.1f cm (prev=%.1f, jump=%.0f%%)",
                                  self.name, raw, self._last_valid_dist, jump_ratio * 100)
                    else:
                        self._accept_reading(raw)
                else:
                    self._accept_reading(raw)

            # Enforce minimum cycle time (60ms for HC-SR04)
            elapsed = time.time() - loop_start
            sleep_t = max(0, _POLL_INTERVAL - elapsed)
            time.sleep(sleep_t)

    def _accept_reading(self, raw: float) -> None:
        """Accept a reading into the median filter and update distance."""
        self._history.append(raw)
        self._last_valid_dist = raw
        with self._lock:
            # Use median filter instead of averaging — much better noise rejection
            if len(self._history) >= 3:
                self._dist = statistics.median(self._history)
            else:
                self._dist = sum(self._history) / len(self._history)
            self._last_valid_time = time.time()

    def _read_once(self) -> Optional[float]:
        """
        Perform a single HC-SR04 distance measurement.

        Timing sequence:
          1. Hold TRIG LOW for ≥5 µs (settle)
          2. Pulse TRIG HIGH for 10 µs
          3. TRIG back to LOW
          4. Wait for ECHO to go HIGH (with timeout)
          5. Measure duration ECHO stays HIGH (with timeout)
          6. Calculate distance = (duration × speed_of_sound) / 2
        """
        try:
            # 1. Pre-trigger settle: ensure TRIG is LOW for at least 5 µs
            GPIO.output(self.trig, GPIO.LOW)
            time.sleep(_SETTLE_TIME)

            # 2. Trigger pulse: 10 µs HIGH
            GPIO.output(self.trig, GPIO.HIGH)
            time.sleep(_TRIG_PULSE)
            GPIO.output(self.trig, GPIO.LOW)

            # 3. Wait for ECHO to go HIGH (sensor sends ultrasonic burst)
            #    Timeout prevents infinite loop if sensor is disconnected
            wait_start = time.time()
            deadline   = wait_start + _TIMEOUT
            max_iters  = 50000  # Safety guard against busy-wait lockup

            pulse_start = time.time()
            iters = 0
            while GPIO.input(self.echo) == 0:
                pulse_start = time.time()
                iters += 1
                if pulse_start > deadline or iters > max_iters:
                    return None

            # 4. Measure echo pulse width (HIGH duration)
            pulse_end = pulse_start
            deadline  = pulse_start + _TIMEOUT
            iters = 0
            while GPIO.input(self.echo) == 1:
                pulse_end = time.time()
                iters += 1
                if pulse_end > deadline or iters > max_iters:
                    return None

            # 5. Calculate distance
            duration = pulse_end - pulse_start
            if duration <= 0:
                return None

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
            time.sleep(0.02)  # Stagger sensor starts to avoid GPIO contention
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

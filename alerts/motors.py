"""
motors.py
─────────
Vibration Motor Controller for Raspberry Pi 4B.

Two DC vibration motors provide haptic feedback in the seat/wheel:
  • Left Motor  → BCM 20 (PWM-capable)
  • Right Motor → BCM 21 (PWM-capable)

Supports:
  • Off         — motor stopped
  • Continuous  — motor always on (full PWM duty)
  • Pulsed      — motor pulses at configured rate (for critical alerts)

Falls back gracefully when RPi.GPIO is unavailable (PC development).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

from config import MOTOR_PINS, WARNING

log = logging.getLogger(__name__)

_PWM_FREQ       = WARNING["pwm_frequency_hz"]
_DUTY_CAUTION   = WARNING["motor_duty_caution"]
_DUTY_CRITICAL  = WARNING["motor_duty_critical"]
_PULSE_PERIOD   = 1.0 / WARNING["motor_pulse_hz"]


# ─────────────────────────────────────────────────────────────────────────────

class VibrationMotor:
    """Single PWM-driven vibration motor."""

    def __init__(self, name: str, pin: int) -> None:
        self.name   = name
        self.pin    = pin
        self._state = "off"
        self._lock  = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pwm   = None
        self._sim_active: bool = False

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self._pwm = GPIO.PWM(pin, _PWM_FREQ)
            self._pwm.start(0)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._sim_active

    def off(self) -> None:
        self._change_state("off")
        self._set_duty(0)

    def pulse(self) -> None:
        """Critical alert: motor pulses on/off at configured rate."""
        self._change_state("pulse")

    def continuous(self, duty: int = _DUTY_CRITICAL) -> None:
        """Continuous vibration at given duty cycle (0–100)."""
        self._change_state("continuous")
        self._set_duty(duty)

    def cleanup(self) -> None:
        self.off()
        if GPIO_AVAILABLE and self._pwm:
            self._pwm.stop()
            GPIO.cleanup(self.pin)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _change_state(self, new_state: str) -> None:
        with self._lock:
            if self._state == new_state:
                return
            self._state = new_state
            if new_state == "pulse":
                if not (self._thread and self._thread.is_alive()):
                    self._thread = threading.Thread(
                        target=self._pulse_loop, name=f"motor-{self.name}", daemon=True
                    )
                    self._thread.start()

    def _pulse_loop(self) -> None:
        half = _PULSE_PERIOD / 2
        while True:
            with self._lock:
                if self._state != "pulse":
                    break
            self._set_duty(_DUTY_CRITICAL)
            time.sleep(half)
            with self._lock:
                if self._state != "pulse":
                    break
            self._set_duty(0)
            time.sleep(half)

    def _set_duty(self, duty: int) -> None:
        self._sim_active = duty > 0
        if GPIO_AVAILABLE and self._pwm:
            self._pwm.ChangeDutyCycle(max(0, min(100, duty)))
        else:
            if duty > 0:
                log.debug("MOTOR [%s] duty=%d%%", self.name, duty)


# ─────────────────────────────────────────────────────────────────────────────

class MotorController:
    """
    Manages both left and right vibration motors.

    Provides semantic methods matching the project specification:
      - ``left_threat()``  → left motor pulses
      - ``right_threat()`` → right motor pulses
      - ``rear_threat()``  → BOTH motors pulse simultaneously
      - ``all_off()``      → stop all

    Usage::

        motors = MotorController()
        motors.left_threat()
        time.sleep(2)
        motors.all_off()
    """

    def __init__(self) -> None:
        self._motors: Dict[str, VibrationMotor] = {
            name: VibrationMotor(name=name, pin=pin)
            for name, pin in MOTOR_PINS.items()
        }
        log.info("MotorController initialised (GPIO=%s).", GPIO_AVAILABLE)

    # ── Semantic Alert Methods ────────────────────────────────────────────────

    def left_threat(self) -> None:
        """Left motor pulses for left collision threat."""
        self._motors["left"].pulse()

    def right_threat(self) -> None:
        """Right motor pulses for right collision threat."""
        self._motors["right"].pulse()

    def rear_threat(self) -> None:
        """Both motors pulse for rear collision threat."""
        self._motors["left"].pulse()
        self._motors["right"].pulse()

    def all_off(self) -> None:
        for m in self._motors.values():
            m.off()

    # ── Generic ───────────────────────────────────────────────────────────────

    def apply(self, position: str, mode: str) -> None:
        """
        Directly control a motor.

        :param position: 'left' | 'right'
        :param mode:     'off' | 'pulse' | 'continuous'
        """
        motor = self._motors.get(position)
        if not motor:
            return
        {"off": motor.off, "pulse": motor.pulse, "continuous": motor.continuous}.get(
            mode, motor.off
        )()

    def get_status(self) -> Dict[str, dict]:
        return {
            name: {"state": m.state, "is_active": m.is_active}
            for name, m in self._motors.items()
        }

    def cleanup(self) -> None:
        for m in self._motors.values():
            m.cleanup()

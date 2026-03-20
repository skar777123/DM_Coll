"""
motors.py
─────────
Vibration Motor Controller for Raspberry Pi 4B.

Two DC vibration motors provide haptic feedback in the seat / wheel:
  • Left Motor  → BCM 20  (PWM-capable)
  • Right Motor → BCM 21  (PWM-capable)

Modes:
  off        → motor stopped
  pulse      → motor cycles on/off at motor_pulse_hz (critical alert)
  continuous → motor always on at given duty cycle

Semantic alert shortcuts:
  left_threat()  → left motor pulses
  right_threat() → right motor pulses
  rear_threat()  → BOTH motors pulse simultaneously

Requires RPi.GPIO. Must run on a Raspberry Pi with real GPIO hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import RPi.GPIO as GPIO   # Real hardware required — must run on Raspberry Pi

from config import MOTOR_PINS, WARNING

log = logging.getLogger(__name__)

_PWM_FREQ      = WARNING["pwm_frequency_hz"]
_DUTY_CAUTION  = WARNING["motor_duty_caution"]
_DUTY_CRITICAL = WARNING["motor_duty_critical"]
_PULSE_PERIOD  = 1.0 / WARNING["motor_pulse_hz"]


# ─────────────────────────────────────────────────────────────────────────────

class VibrationMotor:
    """Single PWM-driven vibration motor."""

    def __init__(self, name: str, pin: int) -> None:
        self.name   = name
        self.pin    = pin
        self._state = "off"     # 'off' | 'pulse' | 'continuous'
        self._lock  = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pwm   = None
        self._sim_active: bool = False

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
        if self._pwm is not None:
            self._pwm.stop()
        try:
            GPIO.cleanup(self.pin)
        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _change_state(self, new_state: str) -> None:
        with self._lock:
            if self._state == new_state:
                return
            self._state = new_state
            if new_state == "pulse":
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(
                        target=self._pulse_loop,
                        name=f"motor-{self.name}",
                        daemon=True,
                    )
                    self._thread.start()

    def _pulse_loop(self) -> None:
        half = _PULSE_PERIOD / 2
        try:
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
        finally:
            self._set_duty(0)

    def _set_duty(self, duty: int) -> None:
        self._sim_active = duty > 0
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(max(0, min(100, duty)))


# ─────────────────────────────────────────────────────────────────────────────

class MotorController:
    """
    Manages both left and right vibration motors.

    Semantic methods:
      - ``left_threat()``  → left motor pulses
      - ``right_threat()`` → right motor pulses
      - ``rear_threat()``  → BOTH motors pulse simultaneously
      - ``all_off()``      → stop all motors

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
        log.info("MotorController initialised.")

    # ── Semantic Alert Methods ────────────────────────────────────────────────

    def left_threat(self) -> None:
        """Left motor pulses for left collision threat."""
        self._motors["left"].pulse()

    def right_threat(self) -> None:
        """Right motor pulses for right collision threat."""
        self._motors["right"].pulse()

    def rear_threat(self) -> None:
        """Both motors pulse simultaneously for rear collision threat."""
        self._motors["left"].pulse()
        self._motors["right"].pulse()

    def all_off(self) -> None:
        for m in self._motors.values():
            m.off()

    # ── Generic Control ───────────────────────────────────────────────────────

    def apply(self, position: str, mode: str) -> None:
        """
        Directly control a motor.

        :param position: 'left' | 'right'
        :param mode:     'off' | 'pulse' | 'continuous'
        """
        motor = self._motors.get(position)
        if motor is None:
            return
            
        if mode == "pulse":
            motor.pulse()
        elif mode == "continuous":
            motor.continuous()
        else:
            motor.off()

    def get_status(self) -> Dict[str, dict]:
        return {
            name: {"state": m.state, "is_active": m.is_active}
            for name, m in self._motors.items()
        }

    def cleanup(self) -> None:
        for m in self._motors.values():
            m.cleanup()

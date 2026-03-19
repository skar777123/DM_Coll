"""
leds.py
───────
LED Warning Controller for Raspberry Pi 4B.

Three LEDs provide directional visual alerts on the dashboard:
  • Left LED  → BCM 19
  • Right LED → BCM 26
  • Rear LED  → BCM 13

Modes:
  off    → LED is off
  solid  → LED is permanently on  (caution zone)
  flash  → LED flashes at led_flash_hz  (critical zone)

Requires RPi.GPIO. Must run on a Raspberry Pi with real GPIO hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import RPi.GPIO as GPIO   # Real hardware required — must run on Raspberry Pi

from config import LED_PINS, WARNING

log = logging.getLogger(__name__)

_FLASH_PERIOD = 1.0 / WARNING["led_flash_hz"]


# ─────────────────────────────────────────────────────────────────────────────

class LED:
    """Single GPIO-driven LED with solid / flashing modes."""

    def __init__(self, name: str, pin: int) -> None:
        self.name   = name
        self.pin    = pin
        self._state = "off"     # 'off' | 'solid' | 'flash'
        self._lock  = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._sim_on: bool = False   # simulated physical state

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_on(self) -> bool:
        """True when LED is physically lit (solid or mid-flash)."""
        return self._sim_on

    def off(self) -> None:
        self._set_state("off")
        self._write(False)

    def solid(self) -> None:
        self._set_state("solid")
        self._write(True)

    def flash(self) -> None:
        self._set_state("flash")

    def cleanup(self) -> None:
        self.off()
        try:
            GPIO.cleanup(self.pin)
        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_state(self, new_state: str) -> None:
        with self._lock:
            if self._state == new_state:
                return
            self._state = new_state
            if new_state == "flash":
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(
                        target=self._flash_loop,
                        name=f"led-{self.name}",
                        daemon=True,
                    )
                    self._thread.start()
            # For 'off' and 'solid', the flash loop exits on its own
            # (it checks self._state every half-period).

    def _flash_loop(self) -> None:
        half = _FLASH_PERIOD / 2
        while True:
            with self._lock:
                if self._state != "flash":
                    break
            self._write(True)
            time.sleep(half)
            with self._lock:
                if self._state != "flash":
                    break
            self._write(False)
            time.sleep(half)

    def _write(self, on: bool) -> None:
        self._sim_on = on
        GPIO.output(self.pin, GPIO.HIGH if on else GPIO.LOW)


# ─────────────────────────────────────────────────────────────────────────────

class LEDController:
    """
    Manages all three directional LEDs.

    Usage::

        leds = LEDController()
        leds.apply("left",  "flash")
        leds.apply("right", "solid")
        leds.apply("rear",  "off")
        status = leds.get_status()
    """

    def __init__(self) -> None:
        self._leds: Dict[str, LED] = {
            name: LED(name=name, pin=pin)
            for name, pin in LED_PINS.items()
        }
        log.info("LEDController initialised.")

    def apply(self, position: str, mode: str) -> None:
        """
        Set LED for a position.

        :param position: 'left' | 'right' | 'rear'
        :param mode:     'off'  | 'solid' | 'flash'
        """
        led = self._leds.get(position)
        if not led:
            log.warning("Unknown LED position: %s", position)
            return
        if mode == "off":
            led.off()
        elif mode == "solid":
            led.solid()
        elif mode == "flash":
            led.flash()
        else:
            log.warning("Unknown LED mode: %s", mode)

    def all_off(self) -> None:
        for led in self._leds.values():
            led.off()

    def get_status(self) -> Dict[str, dict]:
        return {
            name: {"state": led.state, "is_on": led.is_on}
            for name, led in self._leds.items()
        }

    def cleanup(self) -> None:
        for led in self._leds.values():
            led.cleanup()

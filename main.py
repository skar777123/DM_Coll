"""
main.py
───────
Blind Spot Detection System — Main Entry Point.

Startup sequence
────────────────
1.  Configure logging
2.  Start ultrasonic sensor manager  (3 × HC-SR04 background threads)
3.  Start camera manager             (3 × ESP32-CAM HTTP streams)
4.  Initialise LED + motor controllers
5.  Choose threat evaluator          (ML engine if models exist, else rule-based)
6.  Inject all objects into dashboard server
7.  Start dashboard server in a daemon thread
8.  Run main control loop at ~16 Hz

Shutdown
────────
Ctrl-C (SIGINT) or SIGTERM → graceful cleanup of GPIO, serial, and threads.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
import threading
import time

from config import LOGGING, ULTRASONIC, DASHBOARD

# ─────────────────────────────────────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

_log_level = getattr(logging, LOGGING["level"].upper(), logging.INFO)
_formatter  = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = logging.handlers.RotatingFileHandler(
    LOGGING["logfile"],
    maxBytes=LOGGING["max_mb"] * 1024 * 1024,
    backupCount=LOGGING["backups"],
)
_file_handler.setFormatter(_formatter)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

logging.basicConfig(level=_log_level, handlers=[_stream_handler, _file_handler])
log = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────────────────
#  Sub-system Imports  (after logging so they can log correctly)
# ─────────────────────────────────────────────────────────────────────────────

from sensors.ultrasonic   import UltrasonicManager
from sensors.camera       import CameraManager
from alerts.leds          import LEDController
from alerts.motors        import MotorController
from detection.zone_logic import ZoneEvaluator
import dashboard.app as dashboard_app


# ─────────────────────────────────────────────────────────────────────────────
#  Threat Evaluator Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_evaluator(leds: LEDController, motors: MotorController):
    """
    Use UnifiedVehicleThreatEngine (YOLO + LSTM) when models are trained.
    Falls back to the simple rule-based ZoneEvaluator otherwise.
    """
    lstm_path = "ML_Model/saved_models/threat_lstm.pt"
    fnet_path = "ML_Model/saved_models/fusion_net.pt"

    if os.path.exists(lstm_path) or os.path.exists(fnet_path):
        try:
            from ML_Model.vehicle_verifier import UnifiedVehicleThreatEngine
            engine = UnifiedVehicleThreatEngine(leds=leds, motors=motors)
            log.info("UnifiedVehicleThreatEngine loaded ✔  (Two-Stage: YOLO → LSTM active)")
            return engine
        except Exception as exc:
            log.warning(
                "UnifiedVehicleThreatEngine failed to load (%s) — falling back to rule-based.",
                exc,
            )

    log.info("Using rule-based ZoneEvaluator (train ML models to upgrade).")
    return ZoneEvaluator(leds=leds, motors=motors)


# ─────────────────────────────────────────────────────────────────────────────
#  Globals  (needed by shutdown hook)
# ─────────────────────────────────────────────────────────────────────────────

_ultra_manager: UltrasonicManager | None = None
_cam_manager:   CameraManager     | None = None
_led_ctrl:      LEDController     | None = None
_motor_ctrl:    MotorController   | None = None
_running = False


def _shutdown(signum=None, frame=None) -> None:
    """Gracefully stop every subsystem, then exit."""
    global _running
    log.info("Shutdown signal received — stopping all subsystems…")
    _running = False

    if _ultra_manager: _ultra_manager.stop()
    if _cam_manager:   _cam_manager.stop()
    if _led_ctrl:      _led_ctrl.cleanup()
    if _motor_ctrl:    _motor_ctrl.cleanup()
    dashboard_app.stop()

    log.info("All subsystems stopped. Goodbye.")
    sys.exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ─────────────────────────────────────────────────────────────────────────────
#  Console Summary Helper
# ─────────────────────────────────────────────────────────────────────────────

def _log_summary(state, cam_manager=None) -> None:
    from detection.zone_logic import SystemState
    if not isinstance(state, SystemState):
        return

    parts = []
    for d in ("left", "right", "rear"):
        ds = state.get(d)
        if ds:
            if ds.zone == "offline":
                parts.append(f"{d.upper():5s}: OFFLINE")
            else:
                parts.append(f"{d.upper():5s}: {ds.distance_cm:6.1f} cm [{ds.zone.upper()}]")

    cam_parts = []
    if cam_manager:
        frames = cam_manager.get_all_frames()
        for d in ("left", "right", "rear"):
            f = frames.get(d)
            cam_parts.append(f"{d[0].upper()}:{'✔' if f else '✘'}")

    summary = " | ".join(parts)
    if cam_parts:
        summary += "  CAM[" + " ".join(cam_parts) + "]"
    log.info(summary)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _ultra_manager, _cam_manager, _led_ctrl, _motor_ctrl, _running

    log.info("=" * 60)
    log.info("  BlindSpotGuard — Starting Up")
    log.info("  Dashboard → http://localhost:%d", DASHBOARD["port"])
    log.info("=" * 60)

    # ── 1. Ultrasonic Sensors ─────────────────────────────────────────────────
    log.info("[1/5] Initialising ultrasonic sensors…")
    _ultra_manager = UltrasonicManager()
    _ultra_manager.start()
    time.sleep(0.3)   # Allow sensors to stabilise and gather initial readings

    # ── 2. Cameras ───────────────────────────────────────────────────────────
    log.info("[2/5] Initialising cameras…")
    try:
        from sensors.scanner import discover_esp32_cameras
        from config import CAMERA_PORTS
        discovered_urls = discover_esp32_cameras()
        if discovered_urls:
            log.info("Applying dynamically discovered camera URLs…")
            for pos, url in discovered_urls.items():
                if pos in CAMERA_PORTS:
                    CAMERA_PORTS[pos]["url"] = url
                    log.info("  %s → %s", pos, url)
    except Exception as exc:
        log.warning("IP auto-discovery failed: %s", exc)

    _cam_manager = CameraManager()
    _cam_manager.start()
    time.sleep(1.0)   # Give camera threads time to connect and get first frames

    # ── 3. Output Devices ────────────────────────────────────────────────────
    log.info("[3/5] Initialising LEDs and motors…")
    _led_ctrl   = LEDController()
    _motor_ctrl = MotorController()

    # ── 4. Decision Engine ───────────────────────────────────────────────────
    log.info("[4/5] Initialising threat evaluator (ML or rule-based)…")
    evaluator = _build_evaluator(leds=_led_ctrl, motors=_motor_ctrl)

    # ── 5. Dashboard ─────────────────────────────────────────────────────────
    log.info("[5/5] Starting dashboard server on port %d…", DASHBOARD["port"])
    dashboard_app.setup(evaluator=evaluator, camera_manager=_cam_manager)
    dash_thread = threading.Thread(
        target=dashboard_app.run, name="dashboard", daemon=True
    )
    dash_thread.start()
    time.sleep(0.3)
    dashboard_app.start_background_emit()

    # ── Main Control Loop (~16 Hz) ────────────────────────────────────────────
    # HC-SR04 runs at ~16 Hz (60 ms cycle). The main loop matches this rate.
    log.info("All subsystems online. Entering main control loop at ~16 Hz…")
    _running      = True
    interval      = ULTRASONIC["polling_interval_s"]   # 60 ms → ~16 Hz
    _last_log     = 0.0

    while _running:
        loop_start = time.perf_counter()

        ultrasonic_data = _ultra_manager.get_all()
        camera_frames   = _cam_manager.get_all_frames()
        state           = evaluator.evaluate(ultrasonic_data, camera_frames)

        # Throttled console summary — every 2 s
        now = time.time()
        if now - _last_log >= 2.0:
            _log_summary(state, _cam_manager)
            _last_log = now

        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, interval - elapsed))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

"""
main.py
───────
Blind Spot Detection System — Main Entry Point.

Startup sequence
────────────────
1.  Configure logging
2.  Start ultrasonic sensor manager  (3 x HC-SR04 background threads)
3.  Start camera manager             (3 x ESP32-CAM serial streams)
4.  Initialise LED + motor controllers
5.  Initialise zone evaluator        (sensor-fusion decision engine)
6.  Inject all objects into dashboard server
7.  Start dashboard server in background thread
8.  Run main control loop at ~50 Hz

Shutdown
────────
Ctrl+C triggers graceful cleanup of GPIO, serial ports, and threads.
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
_formatter = logging.Formatter(
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

from sensors.ultrasonic   import UltrasonicManager
from sensors.camera       import CameraManager
from alerts.leds        import LEDController
from alerts.motors      import MotorController
from detection.zone_logic import ZoneEvaluator
import dashboard.app as dashboard_app


def _build_evaluator(leds, motors):
    """
    Use UnifiedVehicleThreatEngine (YOLO + LSTM) when models are available.
    Falls back to the simple rule-based ZoneEvaluator otherwise.
    """
    lstm_path = "ML_Model/saved_models/threat_lstm.pt"
    fnet_path = "ML_Model/saved_models/fusion_net.pt"
    models_trained = os.path.exists(lstm_path) or os.path.exists(fnet_path)

    if models_trained:
        try:
            from ML_Model.vehicle_verifier import UnifiedVehicleThreatEngine
            engine = UnifiedVehicleThreatEngine(leds=leds, motors=motors)
            log.info("UnifiedVehicleThreatEngine loaded  ✔  (Two-Stage: YOLO -> LSTM active)")
            return engine
        except Exception as exc:
            log.warning("UnifiedVehicleThreatEngine failed to load (%s) — falling back to rule-based.", exc)

    log.info("Using rule-based ZoneEvaluator (train ML models to upgrade).")
    return ZoneEvaluator(leds=leds, motors=motors)



# ─────────────────────────────────────────────────────────────────────────────
#  Globals (for shutdown hook)
# ─────────────────────────────────────────────────────────────────────────────

_ultra_manager: UltrasonicManager | None = None
_cam_manager:   CameraManager     | None = None
_led_ctrl:      LEDController     | None = None
_motor_ctrl:    MotorController   | None = None
_running        = False


def _shutdown(signum=None, frame=None):
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
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _ultra_manager, _cam_manager, _led_ctrl, _motor_ctrl, _running

    log.info("=" * 60)
    log.info("  BlindSpotGuard — Starting Up")
    log.info("  Dashboard → http://localhost:%d", DASHBOARD["port"])
    log.info("=" * 60)

    # ── 1. Sensors ────────────────────────────────────────────────────────────
    log.info("[1/5] Initialising ultrasonic sensors…")
    _ultra_manager = UltrasonicManager()
    _ultra_manager.start()

    log.info("[2/5] Initialising cameras…")
    try:
        from sensors.scanner import discover_esp32_cameras
        from config import CAMERA_PORTS
        discovered_urls = discover_esp32_cameras()
        if discovered_urls:
            log.info("Applying dynamically discovered camera URLs...")
            for pos, url in discovered_urls.items():
                if pos in CAMERA_PORTS:
                    CAMERA_PORTS[pos]["url"] = url
                    log.info("  %s -> %s", pos, url)
    except Exception as exc:
        log.warning("IP Auto-discovery failed: %s", exc)

    _cam_manager = CameraManager()
    _cam_manager.start()
    time.sleep(0.5)   # Give camera threads time to connect

    # ── 2. Output Devices ─────────────────────────────────────────────────────
    log.info("[3/5] Initialising LEDs and motors…")
    _led_ctrl   = LEDController()
    _motor_ctrl = MotorController()

    # ── 3. Decision Engine ────────────────────────────────────────────────────
    log.info("[4/5] Initialising threat evaluator (ML or rule-based)…")
    evaluator = _build_evaluator(leds=_led_ctrl, motors=_motor_ctrl)

    # ── 4. Dashboard ──────────────────────────────────────────────────────────
    log.info("[5/5] Starting dashboard server on port %d…", DASHBOARD["port"])
    dashboard_app.setup(evaluator=evaluator, camera_manager=_cam_manager)
    dash_thread = threading.Thread(
        target=dashboard_app.run, name="dashboard", daemon=True
    )
    dash_thread.start()
    time.sleep(0.3)
    dashboard_app.start_background_emit()

    # ── 5. Control Loop ───────────────────────────────────────────────────────
    log.info("All subsystems online. Entering main control loop at ~50 Hz…")
    _running   = True
    interval   = ULTRASONIC["polling_interval_s"]   # 20 ms → 50 Hz

    while _running:
        loop_start = time.perf_counter()

        ultrasonic_data = _ultra_manager.get_all()
        camera_frames   = _cam_manager.get_all_frames()
        state           = evaluator.evaluate(ultrasonic_data, camera_frames)

        # Throttled console summary (every 2 s)
        if int(time.time()) % 2 == 0 and (time.time() % 2) < interval:
            _log_summary(state)

        elapsed = time.perf_counter() - loop_start
        sleep_t = max(0, interval - elapsed)
        time.sleep(sleep_t)


def _log_summary(state) -> None:
    from detection.zone_logic import SystemState
    if not isinstance(state, SystemState):
        return
    parts = []
    for d in ("left", "right", "rear"):
        ds = state.get(d)
        if ds:
            parts.append(f"{d.upper():5s}: {ds.distance_cm:6.1f}cm [{ds.zone.upper()}]")
    log.info(" | ".join(parts))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

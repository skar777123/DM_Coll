"""
app.py
──────
Flask + SocketIO live dashboard server.

Serves a real-time web UI showing sensor readings, zone states,
camera feeds, and warning status. Emits updates at ~20 Hz via
WebSocket so the browser displays live data without polling.

Endpoints
─────────
GET  /              → Main dashboard HTML
GET  /api/state     → JSON snapshot of current system state
GET  /api/cameras   → JSON with base64 camera frames
POST /api/override  → Manually trigger a warning (test mode)
WS   /socket.io     → Real-time push events
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from config import DASHBOARD

if TYPE_CHECKING:
    from detection.zone_logic import ZoneEvaluator, SystemState

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "blind-spot-secret-2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared reference injected by main.py after startup
_evaluator: Optional["ZoneEvaluator"] = None
_camera_manager = None
_emit_thread: Optional[threading.Thread] = None
_running = False


# ─────────────────────────────────────────────────────────────────────────────
#  REST Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    if _evaluator is None:
        return jsonify({"error": "system not initialised"}), 503
    return jsonify(_evaluator.state.to_dict())


@app.route("/api/cameras")
def api_cameras():
    if _camera_manager is None:
        return jsonify({"error": "cameras not initialised"}), 503
    frames = _camera_manager.get_all_frames()
    result = {}
    for pos, frame in frames.items():
        result[pos] = frame.to_dict() if frame else {"position": pos, "frame_b64": "", "detections": [], "threat": False}
    return jsonify(result)


@app.route("/api/override", methods=["POST"])
def api_override():
    """Test endpoint — simulate a threat for a given direction."""
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "left")
    zone      = data.get("zone", "critical")
    log.info("Manual override: direction=%s zone=%s", direction, zone)
    socketio.emit("override", {"direction": direction, "zone": zone})
    return jsonify({"ok": True, "direction": direction, "zone": zone})


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket Events
# ─────────────────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    log.info("Dashboard client connected.")
    if _evaluator:
        emit("state_update", _evaluator.state.to_dict())


@socketio.on("disconnect")
def on_disconnect():
    log.info("Dashboard client disconnected.")


@socketio.on("request_frame")
def on_request_frame(data):
    """Client can request a specific camera frame on demand."""
    position = data.get("position", "left")
    if _camera_manager:
        frame = _camera_manager.get_frame(position)
        if frame:
            emit("camera_frame", frame.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
#  Background Emit Loop
# ─────────────────────────────────────────────────────────────────────────────

def _emit_loop():
    """Push system state and camera frames to all clients at ~20 Hz."""
    global _running
    interval = DASHBOARD["emit_rate"]
    frame_interval = 0.1   # cameras at 10 Hz (less bandwidth)
    last_cam = 0.0

    while _running:
        try:
            if _evaluator:
                state_dict = _evaluator.state.to_dict()
                socketio.emit("state_update", state_dict)

            now = time.time()
            if _camera_manager and (now - last_cam) >= frame_interval:
                frames = _camera_manager.get_all_frames()
                for pos, frame in frames.items():
                    if frame and frame.frame_b64:
                        socketio.emit("camera_frame", frame.to_dict())
                last_cam = now

        except Exception as exc:
            log.debug("Emit loop error: %s", exc)

        time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API used by main.py
# ─────────────────────────────────────────────────────────────────────────────

def setup(evaluator: "ZoneEvaluator", camera_manager) -> None:
    """Inject shared objects before starting the server."""
    global _evaluator, _camera_manager
    _evaluator      = evaluator
    _camera_manager = camera_manager


def start_background_emit() -> None:
    """Start the background push thread."""
    global _emit_thread, _running
    _running     = True
    _emit_thread = socketio.start_background_task(_emit_loop)
    log.info("Dashboard emit loop started.")


def run() -> None:
    """Start Flask-SocketIO (blocks). Call from main.py in a thread."""
    log.info(
        "Dashboard starting on http://%s:%d",
        DASHBOARD["host"],
        DASHBOARD["port"],
    )
    socketio.run(
        app,
        host=DASHBOARD["host"],
        port=DASHBOARD["port"],
        debug=DASHBOARD["debug"],
        use_reloader=False,
        log_output=False,
    )


def stop() -> None:
    global _running
    _running = False

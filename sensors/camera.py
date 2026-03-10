"""
camera.py
─────────
ESP32-CAM Serial Stream Reader + Object Detection Pipeline.

Each ESP32-CAM streams JPEG frames over a UART serial port.
A background thread receives the MJPEG stream and passes frames
to an optional YOLOv8 detector running on the Pi.

UART Wiring (GPIO BCM):
  Left   → /dev/ttyS0   (UART0)  — RXD: BCM 15, TXD: BCM 14
  Right  → /dev/ttyAMA4 (UART4)  — RXD: BCM  9, TXD: BCM  8 (Pi4)
  Rear   → /dev/ttyAMA2 (UART2)  — RXD: BCM  1, TXD: BCM  0 (Pi4)

Note: Enable additional UARTs in /boot/config.txt:
      dtoverlay=uart2
      dtoverlay=uart4
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.warning("pyserial not found — cameras in SIMULATION mode.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not found — object detection disabled.")

# OpenCV is optional; fall back gracefully
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from config import CAMERA_PORTS, CAMERA

log = logging.getLogger(__name__)

_SOI = b"\xff\xd8"   # JPEG Start-Of-Image
_EOI = b"\xff\xd9"   # JPEG End-Of-Image


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2

@dataclass
class CameraFrame:
    position:   str                        # 'left' | 'right' | 'rear'
    timestamp:  float = field(default_factory=time.time)
    raw_jpeg:   Optional[bytes] = None
    detections: List[Detection] = field(default_factory=list)
    threat:     bool = False               # True if approaching object found
    frame_b64:  str  = ""                  # Base64 JPEG for dashboard

    def to_dict(self) -> dict:
        return {
            "position":   self.position,
            "timestamp":  self.timestamp,
            "threat":     self.threat,
            "frame_b64":  self.frame_b64,
            "detections": [
                {"label": d.label, "confidence": round(d.confidence, 2)}
                for d in self.detections
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────

class _CameraSimulator:
    """Generates synthetic frames (solid colour + text) when serial unavailable."""

    def __init__(self, position: str) -> None:
        self.position = position
        self._frame_count = 0
        self._colours = {"left": (40, 10, 80), "right": (10, 40, 80), "rear": (10, 80, 40)}

    def next_frame(self) -> CameraFrame:
        self._frame_count += 1
        frame = CameraFrame(position=self.position)

        if CV2_AVAILABLE:
            h, w = CAMERA["frame_height"], CAMERA["frame_width"]
            img = np.zeros((h, w, 3), dtype=np.uint8)
            colour = self._colours.get(self.position, (60, 60, 60))
            img[:] = colour  # solid tint
            cv2.putText(img, f"{self.position.upper()} CAM", (10, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
            cv2.putText(img, f"Frame #{self._frame_count}", (10, h // 2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(img, "SIMULATION MODE", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
            _, buf = cv2.imencode(".jpg", img)
            jpeg = buf.tobytes()
            frame.raw_jpeg  = jpeg
            frame.frame_b64 = base64.b64encode(jpeg).decode()
        return frame


class CameraStream:
    """
    Reads MJPEG frames from one ESP32-CAM over a UART serial port.
    Object detection runs in the same thread (rate-limited to avoid starvation).
    """

    def __init__(self, position: str, port: str, baud: int, label: str,
                 model: Optional["YOLO"] = None) -> None:
        self.position = position
        self.port     = port
        self.baud     = baud
        self.label    = label
        self._model   = model
        self._lock    = threading.Lock()
        self._latest: Optional[CameraFrame] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sim     = _CameraSimulator(position) if not SERIAL_AVAILABLE else None
        self._prev_sizes: List[int] = []     # for approach-speed estimation

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def latest_frame(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._latest

    def start(self) -> "CameraStream":
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name=f"cam-{self.position}", daemon=True
        )
        self._thread.start()
        log.info("[Camera %s] started on %s @ %d baud", self.position, self.port, self.baud)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        if not SERIAL_AVAILABLE:
            self._sim_loop()
            return

        buf = b""
        try:
            with serial.Serial(self.port, self.baud, timeout=1.0) as ser:
                log.info("[Camera %s] serial port open.", self.position)
                while self._running:
                    chunk = ser.read(4096)
                    if not chunk:
                        continue
                    buf += chunk
                    # Extract complete JPEG frames
                    while True:
                        start = buf.find(_SOI)
                        end   = buf.find(_EOI, start + 2) if start != -1 else -1
                        if start == -1 or end == -1:
                            break
                        jpeg   = buf[start: end + 2]
                        buf    = buf[end + 2:]
                        frame  = self._process_jpeg(jpeg)
                        with self._lock:
                            self._latest = frame
        except serial.SerialException as exc:
            log.error("[Camera %s] serial error: %s  → switching to simulator.", self.position, exc)
            self._sim = _CameraSimulator(self.position)
            self._sim_loop()

    def _sim_loop(self) -> None:
        interval = 1.0 / CAMERA["fps"]
        while self._running:
            frame = self._sim.next_frame()
            with self._lock:
                self._latest = frame
            time.sleep(interval)

    def _process_jpeg(self, jpeg: bytes) -> CameraFrame:
        frame = CameraFrame(
            position=self.position,
            raw_jpeg=jpeg,
            frame_b64=base64.b64encode(jpeg).decode(),
        )
        if self._model and CV2_AVAILABLE:
            frame.detections = self._detect(jpeg)
            frame.threat      = self._is_threat(frame.detections)
        return frame

    def _detect(self, jpeg: bytes) -> List[Detection]:
        try:
            arr = np.frombuffer(jpeg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return []
            results = self._model.predict(
                img,
                conf=CAMERA["conf_thresh"],
                classes=None,
                verbose=False,
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label  = r.names[cls_id]
                    if label not in CAMERA["target_classes"]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        bbox=(x1, y1, x2, y2),
                    ))
            return detections
        except Exception as exc:
            log.debug("[Camera %s] detection error: %s", self.position, exc)
            return []

    def _is_threat(self, detections: List[Detection]) -> bool:
        """Heuristic: object is a threat if its bounding box width is growing."""
        if not detections:
            self._prev_sizes.clear()
            return False
        largest = max(detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        size = (largest.bbox[2] - largest.bbox[0]) * (largest.bbox[3] - largest.bbox[1])
        self._prev_sizes.append(size)
        if len(self._prev_sizes) > 5:
            self._prev_sizes.pop(0)
        if len(self._prev_sizes) >= 3:
            growth = self._prev_sizes[-1] - self._prev_sizes[-3]
            if growth > CAMERA["approach_speed_thresh_px"] ** 2:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Owns all three ESP32-CAM streams and (optionally) a shared YOLO model.

    Usage::

        mgr = CameraManager()
        mgr.start()
        frames = mgr.get_all_frames()
        mgr.stop()
    """

    def __init__(self) -> None:
        model: Optional["YOLO"] = None
        if YOLO_AVAILABLE:
            try:
                model = YOLO(CAMERA["yolo_model"])
                log.info("YOLOv8 model loaded: %s", CAMERA["yolo_model"])
            except Exception as exc:
                log.warning("Could not load YOLO model: %s", exc)

        self._streams: Dict[str, CameraStream] = {
            name: CameraStream(
                position=name,
                port=cfg["port"],
                baud=cfg["baud"],
                label=cfg["label"],
                model=model,
            )
            for name, cfg in CAMERA_PORTS.items()
        }

    def start(self) -> "CameraManager":
        for s in self._streams.values():
            s.start()
        return self

    def stop(self) -> None:
        for s in self._streams.values():
            s.stop()

    def get_all_frames(self) -> Dict[str, Optional[CameraFrame]]:
        return {name: s.latest_frame for name, s in self._streams.items()}

    def get_frame(self, position: str) -> Optional[CameraFrame]:
        stream = self._streams.get(position)
        return stream.latest_frame if stream else None

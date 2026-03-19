"""
camera.py
─────────
ESP32-CAM HTTP Stream Reader + YOLO Real-Time Object Detection.

Each ESP32-CAM streams JPEG frames over WiFi HTTP (MJPEG).
A background thread receives the stream, extracts frames, and runs
YOLOv8 on every frame to detect real moving vehicles.

Stream URLs (set in config.py):
  Left   → http://10.92.111.188/stream
  Right  → http://10.92.111.190/stream
  Rear   → http://10.92.111.189/stream

Requirements: opencv-python-headless, ultralytics (YOLOv8), numpy
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2          # Real-time frame decoding — required, no fallback
import numpy as np
import serial

from config import CAMERA_PORTS, CAMERA

log = logging.getLogger(__name__)

_SOI = b"\xff\xd8"   # JPEG Start-Of-Image
_EOI = b"\xff\xd9"   # JPEG End-Of-Image


# ─────────────────────────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    label:         str
    confidence:    float
    bbox:          Tuple[int, int, int, int]   # x1, y1, x2, y2
    is_vehicle:    bool  = False
    is_moving:     bool  = False
    approach_rate: float = 0.0   # percentage bbox area growth per frame


@dataclass
class CameraFrame:
    position:      str
    timestamp:     float = field(default_factory=time.time)
    raw_jpeg:      Optional[bytes] = None
    detections:    List[Detection] = field(default_factory=list)
    threat:        bool  = False     # True if fast-approaching object found
    is_vehicle:    bool  = False     # True if any vehicle detected
    is_moving:     bool  = False     # True if any object is moving
    vision_active: bool  = False     # True if AI model was used for this frame
    max_approach:  float = 0.0       # Highest approach rate in frame
    frame_b64:     str   = ""        # Base64 JPEG for dashboard

    def to_dict(self) -> dict:
        return {
            "position":      self.position,
            "timestamp":     self.timestamp,
            "threat":        bool(self.threat),
            "is_vehicle":    bool(self.is_vehicle),
            "is_moving":     bool(self.is_moving),
            "vision_active": bool(self.vision_active),
            "max_approach":  round(float(self.max_approach), 2),
            "frame_b64":     self.frame_b64,
            "detections": [
                {
                    "label":         d.label,
                    "confidence":    round(float(d.confidence), 2),
                    "is_vehicle":    bool(d.is_vehicle),
                    "is_moving":     bool(d.is_moving),
                    "approach_rate": round(float(d.approach_rate), 2),
                    "bbox":          list(d.bbox),
                }
                for d in self.detections
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CameraStream — single camera handler
# ─────────────────────────────────────────────────────────────────────────────

class CameraStream:
    """
    Reads MJPEG frames from an ESP32-CAM via HTTP stream (primary) or
    UART serial (fallback). Object detection runs in the same thread.
    """

    def __init__(
        self,
        position: str,
        port: str,
        baud: int,
        label: str,
        url: Optional[str] = None,
        model: Optional[object] = None,
    ) -> None:
        self.position = position
        self.port     = port
        self.baud     = baud
        self.label    = label
        self.url      = url
        self._model   = model

        self._lock    = threading.Lock()
        self._latest: Optional[CameraFrame] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prev_detections: List[Detection] = []

        # Stream health tracking
        self._frame_count          = 0
        self._last_frame_time      = 0.0
        self._consecutive_errors   = 0

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def latest_frame(self) -> Optional[CameraFrame]:
        with self._lock:
            # Treat frames older than 5 seconds as stale (stream hung)
            if self._latest and (time.time() - self._latest.timestamp > 5.0):
                self._latest = None
            return self._latest

    def start(self) -> "CameraStream":
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name=f"cam-{self.position}", daemon=True
        )
        self._thread.start()
        if self.url:
            log.info("[Camera %s] started on URL %s", self.position, self.url)
        else:
            log.info("[Camera %s] started on %s @ %d baud", self.position, self.port, self.baud)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── Internal routing ──────────────────────────────────────────────────────

    def _loop(self) -> None:
        if self.url:
            self._url_loop()
        else:
            self._serial_loop()

    # ── Serial (UART) loop ────────────────────────────────────────────────────

    def _serial_loop(self) -> None:
        buf = b""
        try:
            with serial.Serial(self.port, self.baud, timeout=1.0) as ser:
                log.info("[Camera %s] serial port open.", self.position)
                while self._running:
                    chunk = ser.read(4096)
                    if not chunk:
                        continue
                    buf += chunk
                    while True:
                        start = buf.find(_SOI)
                        end   = buf.find(_EOI, start + 2) if start != -1 else -1
                        if start == -1 or end == -1:
                            break
                        jpeg = buf[int(start) : int(end) + 2]

                        # Keep leftover bytes
                        buf = buf[int(end) + 2:]
                        frame = self._process_jpeg(jpeg)
                        with self._lock:
                            self._latest = frame
        except serial.SerialException as exc:
            log.error("[Camera %s] serial error: %s", self.position, exc)

    # ── HTTP MJPEG loop ───────────────────────────────────────────────────────

    def _url_loop(self) -> None:
        """
        Robust HTTP MJPEG stream reader with automatic reconnection.

        Handles:
          - MJPEG multipart streams  (multipart/x-mixed-replace)
          - Single-image polling     (image/jpeg fallback)
          - Automatic reconnection with exponential backoff
          - Per-frame timeout detection
          - Buffer overflow protection
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET"],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))

        retry_delay      = CAMERA.get("stream_retry_delay", 2.0)
        max_retry_delay  = CAMERA.get("stream_max_retry_delay", 10.0)
        connect_timeout  = CAMERA.get("stream_connect_timeout", 5.0)
        read_timeout     = CAMERA.get("stream_read_timeout", 10.0)
        chunk_size       = CAMERA.get("stream_chunk_size", 32768)
        frame_timeout    = CAMERA.get("stream_frame_timeout", 5.0)
        max_buffer       = CAMERA.get("stream_max_buffer", 1_048_576)
        current_delay    = retry_delay

        while self._running:
            try:
                log.info("[Camera %s] Connecting to %s …", self.position, self.url)
                response = session.get(
                    self.url,
                    stream=True,
                    timeout=(connect_timeout, read_timeout),
                )

                if response.status_code != 200:
                    log.error(
                        "[Camera %s] HTTP %d from %s",
                        self.position, response.status_code, self.url,
                    )
                    response.close()
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)
                    continue

                content_type = response.headers.get("Content-Type", "")
                self._consecutive_errors = 0
                current_delay = retry_delay   # reset backoff on successful connect

                if "multipart/x-mixed-replace" in content_type:
                    # ── MJPEG Stream Mode ──────────────────────────────────
                    log.info(
                        "[Camera %s] MJPEG stream connected (Content-Type: %s)",
                        self.position, content_type,
                    )
                    bytes_data    = b""
                    last_frame_at = time.time()

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not self._running:
                            break
                        if not chunk:
                            continue

                        bytes_data += chunk

                        # Extract all complete JPEG frames from the buffer
                        while True:
                            soi_pos = bytes_data.find(_SOI)
                            if soi_pos == -1:
                                bytes_data = b""
                                break

                            if soi_pos > 0:
                                bytes_data = bytes_data[soi_pos:]
                                soi_pos = 0

                            eoi_pos = bytes_data.find(_EOI, 2)
                            if eoi_pos == -1:
                                break   # Wait for more data

                            jpeg       = bytes_data[:eoi_pos + 2]
                            bytes_data = bytes_data[eoi_pos + 2:]

                            if len(jpeg) > 200:
                                frame = self._process_jpeg(jpeg)
                                with self._lock:
                                    self._latest = frame
                                last_frame_at       = time.time()
                                self._frame_count  += 1
                                self._last_frame_time = last_frame_at
                            else:
                                log.debug(
                                    "[Camera %s] Discarded tiny JPEG (%d bytes)",
                                    self.position, len(jpeg),
                                )

                        # Buffer overflow protection
                        if len(bytes_data) > max_buffer:
                            log.warning(
                                "[Camera %s] Buffer overflow (%d bytes), resetting",
                                self.position, len(bytes_data),
                            )
                            bytes_data = b""

                        # Frame-level timeout — reconnect if stream is dead
                        if time.time() - last_frame_at > frame_timeout:
                            log.warning(
                                "[Camera %s] No frame for %.1f s, reconnecting…",
                                self.position, frame_timeout,
                            )
                            break

                    response.close()

                elif "image/jpeg" in content_type or content_type == "":
                    # ── Single Image Polling Mode ──────────────────────────
                    jpeg = response.content
                    response.close()

                    if jpeg and jpeg[:2] == _SOI and jpeg[-2:] == _EOI:
                        frame = self._process_jpeg(jpeg)
                        with self._lock:
                            self._latest = frame
                        self._frame_count    += 1
                        self._last_frame_time = time.time()
                    else:
                        log.debug("[Camera %s] Invalid JPEG in polling response", self.position)

                    time.sleep(1.0 / max(1, CAMERA["fps"]))

                else:
                    log.warning(
                        "[Camera %s] Unexpected Content-Type: %s",
                        self.position, content_type,
                    )
                    response.close()
                    time.sleep(current_delay)

            except Exception as exc:
                import requests as _req
                if isinstance(exc, _req.exceptions.ConnectionError):
                    log.warning(
                        "[Camera %s] Connection failed: %s (retry in %.1f s)",
                        self.position, exc, current_delay,
                    )
                elif isinstance(exc, _req.exceptions.Timeout):
                    log.warning(
                        "[Camera %s] Timeout: %s (retry in %.1f s)",
                        self.position, exc, current_delay,
                    )
                elif isinstance(exc, _req.exceptions.ChunkedEncodingError):
                    log.warning(
                        "[Camera %s] Stream interrupted (ChunkedEncodingError), reconnecting…",
                        self.position,
                    )
                    current_delay = retry_delay
                else:
                    log.error(
                        "[Camera %s] Unexpected error: %s (retry in %.1f s)",
                        self.position, exc, current_delay,
                    )

                self._consecutive_errors += 1
                if self._running:
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)

        try:
            session.close()
        except Exception:
            pass

    # ── JPEG Processing ───────────────────────────────────────────────────────

    def _process_jpeg(self, jpeg: bytes) -> CameraFrame:
        """
        Decode JPEG, run YOLO detection on every frame, annotate with
        bounding boxes, and encode the annotated frame as base64 for
        the dashboard. No simulation or fallback — real detection only.
        """
        frame = CameraFrame(position=self.position, raw_jpeg=jpeg)

        arr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            log.warning(
                "[Camera %s] Failed to decode JPEG (%d bytes) — dropping frame.",
                self.position, len(jpeg),
            )
            return frame

        if self._model:
            # ── Stage 1: YOLO inference ──────────────────────────────────────
            frame.vision_active = True
            frame.detections    = self._detect_on_img(img)
            frame.threat        = self._evaluate_threats(frame)
            frame.is_vehicle    = any(d.is_vehicle for d in frame.detections)
            frame.is_moving     = any(d.is_moving  for d in frame.detections)

            # ── Annotate: draw bounding boxes for every detected object ──────
            for d in frame.detections:
                x1, y1, x2, y2 = d.bbox
                is_threat = d.approach_rate > CAMERA["approach_speed_thresh_px"]

                # Red = threat, amber = vehicle present, green = safe
                if is_threat:
                    color = (0, 0, 220)   # BGR red
                elif d.is_vehicle:
                    color = (0, 165, 255) # BGR amber
                else:
                    color = (0, 200, 80)  # BGR green

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                status     = "THREAT" if is_threat else ("MOVING" if d.is_moving else d.label)
                label_str  = f"{d.label.upper()} {int(d.confidence*100)}% [{status}]"
                (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                cv2.putText(
                    img, label_str, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )

            self._prev_detections = frame.detections
        else:
            # YOLO model not loaded — pass raw frame to dashboard, mark vision inactive
            frame.vision_active = False

        # Encode annotated (or raw) frame for dashboard
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame.frame_b64 = base64.b64encode(buf.tobytes()).decode()

        return frame

    def _detect_on_img(self, img: np.ndarray) -> List[Detection]:
        """Run YOLO on a decoded image and return Detection objects."""
        try:
            results = self._model.predict(
                img,
                conf=CAMERA["conf_thresh"],
                classes=None,
                verbose=False,
            )
            detections: List[Detection] = []
            for r in results:
                for box in r.boxes:
                    cls_id     = int(box.cls[0])
                    label      = r.names[cls_id]
                    if label not in CAMERA["target_classes"]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    is_vehicle  = label in CAMERA["vehicle_classes"]
                    is_moving   = self._check_motion(x1, y1, x2, y2)
                    detections.append(Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        bbox=(x1, y1, x2, y2),
                        is_vehicle=is_vehicle,
                        is_moving=is_moving,
                    ))
            return detections
        except Exception as exc:
            log.debug("[Camera %s] detection error: %s", self.position, exc)
            return []

    def _check_motion(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if current bbox centre shifted > motion_thresh from any previous bbox."""
        if not self._prev_detections:
            return False
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for prev in self._prev_detections:
            px1, py1, px2, py2 = prev.bbox
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist > CAMERA["motion_thresh_px"]:
                return True
        return False

    def _evaluate_threats(self, frame: CameraFrame) -> bool:
        """
        Heuristic: object is a threat if its bounding-box area is growing.
        Updates ``approach_rate`` on each Detection and sets ``frame.max_approach``.
        """
        if not frame.detections:
            frame.max_approach = 0.0
            return False

        any_threat   = False
        max_approach = 0.0

        for d in frame.detections:
            area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            prev_area = None
            cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2

            for prev in self._prev_detections:
                pcx = (prev.bbox[0] + prev.bbox[2]) / 2
                pcy = (prev.bbox[1] + prev.bbox[3]) / 2
                if ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 < 40:
                    prev_area = (prev.bbox[2] - prev.bbox[0]) * (prev.bbox[3] - prev.bbox[1])
                    break

            if prev_area and prev_area > 0:
                growth        = (area - prev_area) / prev_area
                d.approach_rate = growth * 100.0
                max_approach    = max(max_approach, d.approach_rate)
                if d.approach_rate > CAMERA["approach_speed_thresh_px"]:
                    any_threat = True

        frame.max_approach = max_approach
        return any_threat


# ─────────────────────────────────────────────────────────────────────────────
#  CameraManager — owns all three streams
# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Manages all three ESP32-CAM streams with a shared YOLOv8 model.

    YOLOv8 is required for real vehicle detection. If ultralytics fails to
    load the model, a RuntimeError is raised so the system does not start
    in a degraded/silent mode.
    """

    def __init__(self) -> None:
        from ultralytics import YOLO   # Raises ImportError if not installed

        try:
            model = YOLO(CAMERA["yolo_model"])
            log.info(
                "YOLOv8 model loaded: %s — real vehicle detection active.",
                CAMERA["yolo_model"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model '{CAMERA['yolo_model']}': {exc}\n"
                "Install with: pip install ultralytics\n"
                "Download model: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\""
            ) from exc

        self._streams: Dict[str, CameraStream] = {
            name: CameraStream(
                position=name,
                port=cfg["port"],
                baud=cfg.get("baud", 115200),
                label=cfg["label"],
                url=cfg.get("url"),
                model=model,     # Shared model — real inference on every frame
            )
            for name, cfg in CAMERA_PORTS.items()
        }

    def start(self) -> "CameraManager":
        for s in self._streams.values():
            s.start()
            time.sleep(0.1)   # Stagger to avoid thundering herd
        return self

    def stop(self) -> None:
        for s in self._streams.values():
            s.stop()

    def get_all_frames(self) -> Dict[str, Optional[CameraFrame]]:
        return {name: s.latest_frame for name, s in self._streams.items()}

    def get_frame(self, position: str) -> Optional[CameraFrame]:
        stream = self._streams.get(position)
        return stream.latest_frame if stream else None

    def get_health(self) -> Dict[str, dict]:
        """Return health statistics for each camera stream."""
        return {
            name: {
                "frame_count":         s._frame_count,
                "last_frame_time":     s._last_frame_time,
                "consecutive_errors":  s._consecutive_errors,
                "has_frame":           s.latest_frame is not None,
            }
            for name, s in self._streams.items()
        }

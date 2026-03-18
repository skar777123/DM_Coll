"""
camera.py
─────────
ESP32-CAM HTTP Stream Reader + Object Detection Pipeline.

Each ESP32-CAM streams JPEG frames over WiFi HTTP (MJPEG).
A background thread receives the stream and passes frames
to an optional YOLOv8 detector running on the Pi.

Stream URLs (configured in config.py):
  Left   → http://10.132.20.188/stream
  Right  → http://10.132.20.101/stream
  Rear   → http://10.132.20.209/stream

Note: ESP32-CAMs serve multipart/x-mixed-replace MJPEG streams
      on the /stream endpoint when using the standard CameraWebServer sketch.
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

import serial

# Ultralytics and CV2 moved to local imports/checks to avoid SIGILL on ARM
YOLO_AVAILABLE = True # Assume true, check at runtime
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from config import CAMERA_PORTS, CAMERA
from utils_safety import is_module_safe

log = logging.getLogger(__name__)

_SOI = b"\xff\xd8"   # JPEG Start-Of-Image
_EOI = b"\xff\xd9"   # JPEG End-Of-Image


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    is_vehicle: bool = False
    is_moving:  bool = False
    approach_rate: float = 0.0         # % growth per frame

@dataclass
class CameraFrame:
    position:   str                        # 'left' | 'right' | 'rear'
    timestamp:  float = field(default_factory=time.time)
    raw_jpeg:   Optional[bytes] = None
    detections: List[Detection] = field(default_factory=list)
    threat:     bool = False               # True if approaching object found
    is_vehicle: bool = False               # True if any vehicle detected
    is_moving:  bool = False               # True if any object is moving
    vision_active: bool = False            # True if AI model was used for this frame
    max_approach: float = 0.0              # Highest approach rate in frame
    frame_b64:  str  = ""                  # Base64 JPEG for dashboard

    def to_dict(self) -> dict:
        return {
            "position":     self.position,
            "timestamp":    self.timestamp,
            "threat":       self.threat,
            "is_vehicle":   self.is_vehicle,
            "is_moving":    self.is_moving,
            "vision_active": self.vision_active,
            "max_approach": round(self.max_approach, 2),
            "frame_b64":    self.frame_b64,
            "detections": [
                {
                    "label": d.label, 
                    "confidence": round(d.confidence, 2),
                    "is_vehicle": d.is_vehicle,
                    "is_moving": d.is_moving,
                    "approach_rate": round(d.approach_rate, 2),
                    "bbox": d.bbox
                }
                for d in self.detections
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────

class CameraStream:
    """
    Reads MJPEG frames from an ESP32-CAM via HTTP Stream (primary) or UART serial (fallback).
    Object detection runs in the same thread (rate-limited to avoid starvation).
    """

    def __init__(self, position: str, port: str, baud: int, label: str,
                 url: Optional[str] = None, model: Optional["YOLO"] = None) -> None:
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
        self._prev_sizes: List[int] = []     # for approach-speed estimation
        self._prev_detections: List[Detection] = []
        
        # Stream health tracking
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._consecutive_errors = 0

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def latest_frame(self) -> Optional[CameraFrame]:
        with self._lock:
            # If the current frame is older than 5 seconds, the stream has hung/disconnected
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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        if self.url:
            self._url_loop()
            return

        # Serial mode (UART fallback)
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
            log.error("[Camera %s] serial error: %s", self.position, exc)

    def _url_loop(self) -> None:
        """
        Robust HTTP MJPEG stream reader with automatic reconnection.
        
        Handles:
          - MJPEG multipart streams (multipart/x-mixed-replace)
          - Single image polling (image/jpeg fallback)
          - Automatic reconnection with exponential backoff
          - Per-frame timeout detection
          - Buffer overflow protection
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        # Setup a persistent session with retries
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET"],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))

        retry_delay = CAMERA.get("stream_retry_delay", 2.0)
        max_retry_delay = CAMERA.get("stream_max_retry_delay", 10.0)
        connect_timeout = CAMERA.get("stream_connect_timeout", 5.0)
        read_timeout = CAMERA.get("stream_read_timeout", 10.0)
        chunk_size = CAMERA.get("stream_chunk_size", 32768)
        frame_timeout = CAMERA.get("stream_frame_timeout", 5.0)
        max_buffer = CAMERA.get("stream_max_buffer", 1048576)
        current_delay = retry_delay

        while self._running:
            try:
                log.info("[Camera %s] Connecting to %s ...", self.position, self.url)
                
                response = session.get(
                    self.url,
                    stream=True,
                    timeout=(connect_timeout, read_timeout),
                )
                
                if response.status_code != 200:
                    log.error("[Camera %s] HTTP %d from %s", self.position, response.status_code, self.url)
                    response.close()
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)
                    continue

                content_type = response.headers.get("Content-Type", "")
                self._consecutive_errors = 0
                current_delay = retry_delay  # Reset backoff on successful connect

                if "multipart/x-mixed-replace" in content_type:
                    # ── MJPEG Stream Mode ──
                    log.info("[Camera %s] MJPEG stream connected (Content-Type: %s)", 
                             self.position, content_type)
                    
                    bytes_data = b""
                    last_frame_at = time.time()

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not self._running:
                            break
                        
                        if not chunk:
                            continue
                            
                        bytes_data += chunk

                        # Extract all complete JPEG frames from the buffer
                        frames_found = 0
                        while True:
                            soi_pos = bytes_data.find(_SOI)
                            if soi_pos == -1:
                                # No SOI marker found — discard everything
                                bytes_data = b""
                                break
                                
                            # Discard any data before the first SOI
                            if soi_pos > 0:
                                bytes_data = bytes_data[soi_pos:]
                                soi_pos = 0
                            
                            eoi_pos = bytes_data.find(_EOI, 2)  # Search after SOI
                            if eoi_pos == -1:
                                # No complete frame yet — wait for more data
                                break

                            # Extract the complete JPEG
                            jpeg = bytes_data[:eoi_pos + 2]
                            bytes_data = bytes_data[eoi_pos + 2:]

                            # Validate JPEG integrity (minimum reasonable size)
                            if len(jpeg) > 200:
                                frame = self._process_jpeg(jpeg)
                                with self._lock:
                                    self._latest = frame
                                last_frame_at = time.time()
                                self._frame_count += 1
                                self._last_frame_time = last_frame_at
                                frames_found += 1
                            else:
                                log.debug("[Camera %s] Discarded tiny JPEG (%d bytes)",
                                          self.position, len(jpeg))

                        # Buffer overflow protection
                        if len(bytes_data) > max_buffer:
                            log.warning("[Camera %s] Buffer overflow (%d bytes), resetting",
                                        self.position, len(bytes_data))
                            bytes_data = b""

                        # Frame-level timeout: if we haven't received a complete frame
                        # in N seconds, the stream is probably dead — reconnect
                        if time.time() - last_frame_at > frame_timeout:
                            log.warning("[Camera %s] No frame for %.1fs, reconnecting...",
                                        self.position, frame_timeout)
                            break

                    response.close()

                elif "image/jpeg" in content_type or content_type == "":
                    # ── Single Image Polling Mode ──
                    jpeg = response.content
                    response.close()
                    
                    if jpeg and jpeg[:2] == _SOI and jpeg[-2:] == _EOI:
                        frame = self._process_jpeg(jpeg)
                        with self._lock:
                            self._latest = frame
                        self._frame_count += 1
                        self._last_frame_time = time.time()
                    else:
                        log.debug("[Camera %s] Invalid JPEG in polling response", self.position)

                    # Polling interval based on configured FPS
                    time.sleep(1.0 / max(1, CAMERA["fps"]))
                    
                else:
                    log.warning("[Camera %s] Unexpected Content-Type: %s", 
                                self.position, content_type)
                    response.close()
                    time.sleep(current_delay)

            except requests.exceptions.ConnectionError as exc:
                log.warning("[Camera %s] Connection failed: %s (retrying in %.1fs)",
                            self.position, exc, current_delay)
                self._consecutive_errors += 1
                if self._running:
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)

            except requests.exceptions.Timeout as exc:
                log.warning("[Camera %s] Timeout: %s (retrying in %.1fs)",
                            self.position, exc, current_delay)
                self._consecutive_errors += 1
                if self._running:
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)

            except requests.exceptions.ChunkedEncodingError:
                log.warning("[Camera %s] Stream interrupted (ChunkedEncodingError), reconnecting...",
                            self.position)
                self._consecutive_errors += 1
                if self._running:
                    time.sleep(retry_delay)

            except Exception as exc:
                log.error("[Camera %s] Unexpected error: %s (retrying in %.1fs)",
                           self.position, exc, current_delay)
                self._consecutive_errors += 1
                if self._running:
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)

        try:
            session.close()
        except Exception:
            pass

    def _process_jpeg(self, jpeg: bytes) -> CameraFrame:
        frame = CameraFrame(
            position=self.position,
            raw_jpeg=jpeg,
        )
        
        if CV2_AVAILABLE:
            # Decode for processing/drawing
            arr = np.frombuffer(jpeg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if img is not None:
                if self._model:
                    frame.vision_active = True
                    # 1. Detect
                    frame.detections = self._detect_on_img(img)
                    
                    # 2. Evaluate threats & motion
                    frame.threat      = self._evaluate_threats(frame)
                    frame.is_vehicle  = any(d.is_vehicle for d in frame.detections)
                    frame.is_moving   = any(d.is_moving for d in frame.detections)
                    
                    # 3. Draw on image
                    for d in frame.detections:
                        # Only frame vehicles as requested
                        if not d.is_vehicle:
                            continue

                        x1, y1, x2, y2 = d.bbox
                        
                        # RED if it's a threat (fast approach or critical), GREEN if safe
                        is_threat = d.approach_rate > CAMERA["approach_speed_thresh_px"]
                        color = (0, 0, 255) if is_threat else (0, 255, 0) # BGR
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        status_label = "THREAT" if is_threat else "SAFE"
                        label_str = f"{d.label.upper()} ({status_label})"
                        
                        # Draw label background
                        (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
                        cv2.putText(img, label_str, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Update prev detections for next frame
                    self._prev_detections = frame.detections

                # 4. Encode back to base64 for dashboard
                _, buf = cv2.imencode(".jpg", img)
                frame.frame_b64 = base64.b64encode(buf.tobytes()).decode()
            else:
                # Fallback if decode fails but we have raw jpeg
                frame.frame_b64 = base64.b64encode(jpeg).decode()
        else:
            frame.frame_b64 = base64.b64encode(jpeg).decode()
            
        return frame

    def _detect_on_img(self, img: np.ndarray) -> List[Detection]:
        """Same as _detect but takes an already decoded image."""
        try:
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
                    
                    is_vehicle = label in CAMERA["vehicle_classes"]
                    is_moving  = self._check_motion(x1, y1, x2, y2)
                    
                    detections.append(Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        bbox=(x1, y1, x2, y2),
                        is_vehicle=is_vehicle,
                        is_moving=is_moving
                    ))
            return detections
        except Exception as exc:
            log.debug("[Camera %s] detection error: %s", self.position, exc)
            return []

    def _check_motion(self, x1, y1, x2, y2) -> bool:
        """Checks if current box center has shifted significantly from any previous box."""
        if not self._prev_detections:
            return False
        
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        for prev in self._prev_detections:
            px1, py1, px2, py2 = prev.bbox
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            
            dist = ((cx - pcx)**2 + (cy - pcy)**2)**0.5
            if dist > CAMERA["motion_thresh_px"]:
                return True
        return False

    def _evaluate_threats(self, frame: CameraFrame) -> bool:
        """
        Heuristic: object is a threat if its bounding box width is growing.
        Calculates approach_rate for each detection and updates frame.max_approach.
        """
        if not frame.detections:
            self._prev_sizes.clear()
            frame.max_approach = 0.0
            return False

        any_threat = False
        max_approach = 0.0

        for d in frame.detections:
            # Current area
            area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            
            # Find matching detection in previous frame to calculate growth
            prev_area = None
            cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
            
            for prev in self._prev_detections:
                pcx, pcy = (prev.bbox[0] + prev.bbox[2]) / 2, (prev.bbox[1] + prev.bbox[3]) / 2
                dist = ((cx - pcx)**2 + (cy - pcy)**2)**0.5
                if dist < 40: # Match if centers are close
                    prev_area = (prev.bbox[2] - prev.bbox[0]) * (prev.bbox[3] - prev.bbox[1])
                    break
            
            if prev_area and prev_area > 0:
                growth = (area - prev_area) / prev_area
                # Smooth growth estimate or use as is
                d.approach_rate = growth * 100.0 # percentage growth
                if d.approach_rate > max_approach:
                    max_approach = d.approach_rate
                
                # If area is growing significantly, it's a threat
                if d.approach_rate > CAMERA["approach_speed_thresh_px"]:
                    any_threat = True
        
        frame.max_approach = max_approach
        return any_threat


# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Owns all three ESP32-CAM streams and (optionally) a shared YOLO model.
    """

    def __init__(self) -> None:
        model: Optional["YOLO"] = None
        
        # We try to load YOLO directly. If it fails (e.g. SIGILL or missing), 
        # the system will fall back to basic distance-based alerts.
        try:
            from ultralytics import YOLO
            model = YOLO(CAMERA["yolo_model"])
            log.info("YOLOv8 model loaded: %s", CAMERA["yolo_model"])
        except Exception as exc:
            log.warning("YOLO (ultralytics) is unavailable or failed to load — disabling AI vision: %s", exc)

        self._streams: Dict[str, CameraStream] = {
            name: CameraStream(
                position=name,
                port=cfg["port"],
                baud=cfg.get("baud", 115200),
                label=cfg["label"],
                url=cfg.get("url"),
                model=model,
            )
            for name, cfg in CAMERA_PORTS.items()
        }


    def start(self) -> "CameraManager":
        for s in self._streams.values():
            s.start()
            time.sleep(0.1)  # Stagger starts to avoid thundering herd
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
        """Return health stats for each camera stream."""
        return {
            name: {
                "frame_count": s._frame_count,
                "last_frame_time": s._last_frame_time,
                "consecutive_errors": s._consecutive_errors,
                "has_frame": s.latest_frame is not None,
            }
            for name, s in self._streams.items()
        }

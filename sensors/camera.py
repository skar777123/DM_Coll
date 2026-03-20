"""
camera.py
─────────
ESP32-CAM HTTP Stream Reader + YOLO Real-Time Object Detection.

Each ESP32-CAM streams JPEG frames over WiFi HTTP (MJPEG).
A background thread receives the stream, extracts frames, and
a separate processing thread runs YOLO on the latest frame.

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

import cv2
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
    Reads MJPEG frames from an ESP32-CAM via HTTP stream.
    Decouples frame reading from YOLO inference to avoid buffer overflows.
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
        
        self._raw_lock = threading.Lock()
        self._latest_raw_jpeg: Optional[bytes] = None
        self._new_raw_event = threading.Event()
        
        self._running = False
        self._read_thread: Optional[threading.Thread] = None
        self._proc_thread: Optional[threading.Thread] = None
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
        self._read_thread = threading.Thread(
            target=self._read_loop, name=f"cam-read-{self.position}", daemon=True
        )
        self._read_thread.start()
        
        self._proc_thread = threading.Thread(
            target=self._process_loop, name=f"cam-proc-{self.position}", daemon=True
        )
        self._proc_thread.start()
        
        if self.url:
            log.info("[Camera %s] started on URL %s", self.position, self.url)
        else:
            log.info("[Camera %s] started on %s @ %d baud", self.position, self.port, self.baud)
        return self

    def stop(self) -> None:
        self._running = False
        self._new_raw_event.set()
        if self._read_thread:
            self._read_thread.join(timeout=2.0)
        if self._proc_thread:
            self._proc_thread.join(timeout=2.0)

    # ── Internal Loops ────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        if self.url:
            self._url_loop()
        else:
            self._serial_loop()

    def _process_loop(self) -> None:
        """Wait for new raw JPEGs and run YOLO detection on them."""
        while self._running:
            if not self._new_raw_event.wait(timeout=1.0):
                continue
            self._new_raw_event.clear()
            
            with self._raw_lock:
                jpeg = self._latest_raw_jpeg
                if not jpeg:
                    continue
            
            frame = self._process_jpeg(jpeg)
            with self._lock:
                self._latest = frame

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
                        buf = buf[int(end) + 2:]
                        
                        if len(jpeg) > 200:
                            with self._raw_lock:
                                self._latest_raw_jpeg = jpeg
                            self._new_raw_event.set()
                            
        except serial.SerialException as exc:
            log.error("[Camera %s] serial error: %s", self.position, exc)

    # ── HTTP MJPEG loop ───────────────────────────────────────────────────────

    def _url_loop(self) -> None:
        """Fast HTTP MJPEG stream reader. Only extracts raw frames."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))

        retry_delay     = CAMERA.get("stream_retry_delay", 2.0)
        max_retry_delay = CAMERA.get("stream_max_retry_delay", 10.0)
        connect_timeout = CAMERA.get("stream_connect_timeout", 5.0)
        read_timeout    = CAMERA.get("stream_read_timeout", 10.0)
        chunk_size      = CAMERA.get("stream_chunk_size", 32768)
        frame_timeout   = CAMERA.get("stream_frame_timeout", 5.0)
        max_buffer      = CAMERA.get("stream_max_buffer", 1_048_576)
        current_delay   = retry_delay

        while self._running:
            try:
                log.info("[Camera %s] Connecting to %s …", self.position, self.url)
                response = session.get(
                    self.url, stream=True, timeout=(connect_timeout, read_timeout)
                )

                if response.status_code != 200:
                    log.error("[Camera %s] HTTP %d from %s", self.position, response.status_code, self.url)
                    response.close()
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)
                    continue

                content_type = response.headers.get("Content-Type", "")
                self._consecutive_errors = 0
                current_delay = retry_delay

                if "multipart/x-mixed-replace" in content_type:
                    log.info("[Camera %s] MJPEG stream connected", self.position)
                    bytes_data    = b""
                    last_frame_at = time.time()

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not self._running: break
                        if not chunk: continue
                        bytes_data += chunk

                        while True:
                            soi_pos = bytes_data.find(_SOI)
                            if soi_pos == -1:
                                if len(bytes_data) > 0:
                                    bytes_data = bytes_data[-1:]
                                break
                            if soi_pos > 0:
                                bytes_data = bytes_data[soi_pos:]
                                soi_pos = 0

                            eoi_pos = bytes_data.find(_EOI, 2)
                            if eoi_pos == -1: break

                            jpeg       = bytes_data[:eoi_pos + 2]
                            bytes_data = bytes_data[eoi_pos + 2:]

                            if len(jpeg) > 200:
                                with self._raw_lock:
                                    self._latest_raw_jpeg = jpeg
                                self._new_raw_event.set()
                                
                                last_frame_at = time.time()
                                self._frame_count += 1
                                self._last_frame_time = last_frame_at

                        if len(bytes_data) > max_buffer:
                            bytes_data = b""

                        if time.time() - last_frame_at > frame_timeout:
                            log.warning("[Camera %s] Frame timeout, reconnecting…", self.position)
                            break
                    response.close()

                elif "image/jpeg" in content_type or content_type == "":
                    jpeg = response.content
                    response.close()
                    if jpeg and jpeg[:2] == _SOI and jpeg[-2:] == _EOI:
                        with self._raw_lock:
                            self._latest_raw_jpeg = jpeg
                        self._new_raw_event.set()
                        self._frame_count += 1
                        self._last_frame_time = time.time()
                    time.sleep(1.0 / max(1, CAMERA["fps"]))
                else:
                    response.close()
                    time.sleep(current_delay)

            except Exception as exc:
                log.warning("[Camera %s] Stream error: %s", self.position, exc)
                self._consecutive_errors += 1
                if self._running:
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 1.5, max_retry_delay)

        try: session.close()
        except: pass

    # ── JPEG Processing ───────────────────────────────────────────────────────

    def _process_jpeg(self, jpeg: bytes) -> CameraFrame:
        """Decode JPEG, run YOLO, annotate, and encode as base64."""
        frame = CameraFrame(position=self.position, raw_jpeg=jpeg)

        arr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return frame

        if self._model:
            frame.vision_active = True
            frame.detections    = self._detect_on_img(img)
            frame.threat        = self._evaluate_threats(frame)
            frame.is_vehicle    = any(d.is_vehicle for d in frame.detections)
            frame.is_moving     = any(d.is_moving  for d in frame.detections)

            for d in frame.detections:
                x1, y1, x2, y2 = d.bbox
                is_threat = d.approach_rate > CAMERA["approach_speed_thresh_px"]
                color = (0, 0, 220) if is_threat else ((0, 165, 255) if d.is_vehicle else (0, 200, 80))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                status = "THREAT" if is_threat else ("MOVING" if d.is_moving else d.label)
                label_str = f"{d.label.upper()} {int(d.confidence*100)}% [{status}]"
                (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                cv2.putText(img, label_str, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            self._prev_detections = frame.detections
        else:
            frame.vision_active = False

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame.frame_b64 = base64.b64encode(buf.tobytes()).decode()

        return frame

    def _detect_on_img(self, img: np.ndarray) -> List[Detection]:
        try:
            results = self._model.predict(img, conf=CAMERA["conf_thresh"], classes=None, verbose=False)
            detections: List[Detection] = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    if label not in CAMERA["target_classes"]: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(Detection(
                        label=label, confidence=float(box.conf[0]),
                        bbox=(x1, y1, x2, y2), is_vehicle=(label in CAMERA["vehicle_classes"]),
                        is_moving=self._check_motion(x1, y1, x2, y2)
                    ))
            return detections
        except: return []

    def _check_motion(self, x1, y1, x2, y2) -> bool:
        if not self._prev_detections: return False
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for prev in self._prev_detections:
            px1, py1, px2, py2 = prev.bbox
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            if ((cx - pcx)**2 + (cy - pcy)**2)**0.5 > CAMERA["motion_thresh_px"]: return True
        return False

    def _evaluate_threats(self, frame: CameraFrame) -> bool:
        if not frame.detections:
            frame.max_approach = 0.0
            return False
        any_threat = False
        max_approach = 0.0
        for d in frame.detections:
            area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            prev_area = None
            cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
            for prev in self._prev_detections:
                pcx, pcy = (prev.bbox[0] + prev.bbox[2]) / 2, (prev.bbox[1] + prev.bbox[3]) / 2
                if ((cx - pcx)**2 + (cy - pcy)**2)**0.5 < 40:
                    prev_area = (prev.bbox[2] - prev.bbox[0]) * (prev.bbox[3] - prev.bbox[1])
                    break
            if prev_area and prev_area > 0:
                growth = (area - prev_area) / prev_area
                d.approach_rate = growth * 100.0
                max_approach = max(max_approach, d.approach_rate)
                if d.approach_rate > CAMERA["approach_speed_thresh_px"]: any_threat = True
        frame.max_approach = max_approach
        return any_threat


# ─────────────────────────────────────────────────────────────────────────────
#  CameraManager — owns all three streams
# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    def __init__(self) -> None:
        from ultralytics import YOLO
        try:
            model = YOLO(CAMERA["yolo_model"])
            log.info("YOLOv8 model loaded: %s", CAMERA["yolo_model"])
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model: {exc}")

        self._streams: Dict[str, CameraStream] = {
            name: CameraStream(
                position=name, port=cfg["port"], baud=cfg.get("baud", 115200),
                label=cfg["label"], url=cfg.get("url"), model=model
            )
            for name, cfg in CAMERA_PORTS.items()
        }

    def start(self) -> "CameraManager":
        for s in self._streams.values():
            s.start()
            time.sleep(0.1)
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
        return {
            name: {
                "frame_count": s._frame_count,
                "last_frame_time": s._last_frame_time,
                "consecutive_errors": s._consecutive_errors,
                "has_frame": s.latest_frame is not None,
            }
            for name, s in self._streams.items()
        }

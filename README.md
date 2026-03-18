# Blind Spot Detection System — Project README

# BlindSpotGuard 🛡️
> Real-time, multi-modal blind spot detection for bikes and vehicles.
> **Raspberry Pi 4B · 3× ESP32-CAM · 3× HC-SR04 · YOLOv8**

---

## Hardware

### ESP32-CAM (UART Serial)
| Position | UART       | U0T → Pi RX         | U0R → Pi TX         |
|----------|------------|---------------------|---------------------|
| Left     | UART0      | Pin 10 (BCM 15)     | Pin 8 (BCM 14)      |
| Right    | UART4      | Pin 21 (BCM 9)      | Pin 24 (BCM 8)      |
| Rear     | UART2      | Pin 28 (BCM 1)      | Pin 27 (BCM 0)      |

> Power cameras from **external 5 V** — never from the Pi.

### HC-SR04 Ultrasonic Sensors
| Position      | TRIG (BCM) | ECHO (BCM) | Note                     |
|---------------|------------|------------|--------------------------|
| Left Blindspot| BCM 23     | BCM 24     | ECHO → voltage divider   |
| Right Blindspot| BCM 17    | BCM 27     | ECHO → voltage divider   |
| Rear          | BCM 5      | BCM 6      | ECHO → voltage divider   |

### Output Devices
| Device        | BCM Pin |
|---------------|---------|
| Left LED      | BCM 19  |
| Right LED     | BCM 26  |
| Rear LED      | BCM 13  |
| Left Motor    | BCM 20  |
| Right Motor   | BCM 21  |

---

## Software Architecture

```
main.py                   ← Entry point (50 Hz control loop)
config.py                 ← All pin maps, thresholds, settings

sensors/
  ultrasonic.py           ← HC-SR04 manager (3 background threads)
  camera.py               ← ESP32-CAM MJPEG + YOLOv8 detection

alerts/
  leds.py                 ← LED controller (solid / flash)
  motors.py               ← Vibration motor PWM controller

detection/
  zone_logic.py           ← Sensor-fusion decision engine

dashboard/
  app.py                  ← Flask + SocketIO server
  templates/index.html    ← Live web dashboard
  static/css/style.css
  static/js/dashboard.js

tests/
  test_ultrasonic.py
  test_zone_logic.py
```

---

## Detection Zones

| Zone      | Distance     | Camera    | LED      | Motor    |
|-----------|-------------|-----------|----------|----------|
| 🟢 Safe   | > 300 cm    | No threat | Off      | Off      |
| 🟡 Caution| 80–300 cm   | OR threat | Solid    | Off      |
| 🔴 Critical| < 80 cm   | confirmed | Flashing | Pulsing  |

**Rear critical** → **both** motors pulse simultaneously.

---

## Raspberry Pi Setup

### 1. Enable additional UARTs
Add to `/boot/config.txt`:
```
dtoverlay=uart2
dtoverlay=uart4
```
Reboot: `sudo reboot`

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 model (optional)
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 4. Run
```bash
python main.py
```
Dashboard → `http://<pi-ip>:5000`

---

## Running on PC (Simulation Mode)
When `RPi.GPIO` and `pyserial` are not available, all sensors and outputs
run in simulation mode automatically. The dashboard still shows live
animated data.

```bash
pip install flask flask-socketio eventlet numpy opencv-python
python main.py
```
Open `http://localhost:5000`

---

## Running Tests
```bash
python -m pytest tests/ -v
# or individually:
python tests/test_zone_logic.py
python tests/test_ultrasonic.py
```

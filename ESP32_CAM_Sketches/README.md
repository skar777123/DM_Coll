# ESP32-CAM Sketches — BlindSpotGuard

Three separate Arduino sketches, one per camera position.
All share `blindspot_config.h` which contains all endpoint handlers,
camera init, WiFi setup, and mDNS — linked to the Python backend.

---

## Directory Structure

```
ESP32_CAM_Sketches/
├── blindspot_config.h              ← Shared code (handlers, camera init, WiFi)
├── blindspot_cam_LEFT/
│   └── blindspot_cam_LEFT.ino     ← Flash to LEFT camera  → 10.92.111.188
├── blindspot_cam_RIGHT/
│   └── blindspot_cam_RIGHT.ino    ← Flash to RIGHT camera → 10.92.111.190
└── blindspot_cam_REAR/
    └── blindspot_cam_REAR.ino     ← Flash to REAR camera  → 10.92.111.189
```

---

## Quick Start

### 1. Edit WiFi credentials in `blindspot_config.h`
```cpp
#define WIFI_SSID  "YOUR_WIFI_NAME"
#define WIFI_PASS  "YOUR_WIFI_PASSWORD"
```

### 2. Verify the gateway IP in each `.ino` file
```cpp
IPAddress GATEWAY_IP(192, 168, 1, 1);   // ← Change to your router IP
```
Find it with: `ip route | grep default` (Linux/Pi) or `ipconfig` (Windows)

### 3. Open the sketch folder in Arduino IDE
- Open `blindspot_cam_LEFT/blindspot_cam_LEFT.ino`
- Both files (`blindspot_config.h` and the `.ino`) must be in the **same folder**
- Arduino IDE will automatically include the `.h` file

### 4. Arduino IDE board settings
| Setting | Value |
|---|---|
| Board | AI Thinker ESP32-CAM |
| Partition Scheme | Huge APP (3MB No OTA / 1MB SPIFFS) |
| CPU Frequency | 240MHz |
| Flash Mode | QIO |
| Flash Size | 4MB |
| Upload Speed | 115200 |

### 5. Flash each camera
- Flash LEFT sketch  → LEFT  ESP32-CAM
- Flash RIGHT sketch → RIGHT ESP32-CAM
- Flash REAR sketch  → REAR  ESP32-CAM

---

## HTTP Endpoints (port 80)

| Endpoint | Used by | Description |
|---|---|---|
| `GET /stream` | `sensors/camera.py` | MJPEG multipart stream (15 FPS, QVGA 320×240) |
| `GET /capture` | `sensors/camera.py` | Single JPEG (polling fallback) |
| `GET /id` | `sensors/scanner.py` | Identity JSON `{"position":"left","ip":"..."}` |
| `GET /health` | `dashboard/app.py` | Liveness `{"status":"ok","uptime_ms":...}` |

---

## How Auto-Discovery Works

```
Raspberry Pi (main.py)
    └─ scanner.py:discover_esp32_cameras()
           1. Get local Pi IP  →  e.g. 192.168.1.100
           2. Derive subnet    →  192.168.1.0/24
           3. Exclude: Pi IP, gateway, network, broadcast
           4. nmap -p 80 --open → finds all devices with port 80 open
           5. For each device: GET /id  →  {"position": "left"|"right"|"rear"}
           6. Assign stream URL → config.py CAMERA_PORTS[position]["url"]
```

If static IPs are set correctly in both the sketches and `config.py`, the
system will find cameras immediately. nmap discovery is a fallback for when
IPs change or are unknown.

---

## Verifying a Camera Works

After flashing and powering on, open Serial Monitor (115200 baud).
You should see:
```
═══════════════════════════════════════════════
  BlindSpotGuard  |  Camera: left
  Static IP: 10.92.111.188
═══════════════════════════════════════════════
[CAM] PSRAM found — high-quality mode
[WiFi] Connected!
[WiFi]   IP:      10.92.111.188
[HTTP] Server started on port 80
[HTTP]   /stream   → MJPEG feed (15 FPS, QVGA)
[mDNS] Registered as http://blindspot-left.local
  MJPEG stream : http://10.92.111.188/stream
  Identity JSON: http://10.92.111.188/id
═══════════════════════════════════════════════
```

Test in browser:
- `http://10.92.111.188/id` → should return `{"position":"left",...}`
- `http://10.92.111.188/stream` → should show live video

---

## Matching Python Config

`config.py` on the Raspberry Pi must match:
```python
CAMERA_PORTS = {
    "left":  { "url": "http://10.92.111.188/stream", ... },
    "right": { "url": "http://10.92.111.190/stream", ... },
    "rear":  { "url": "http://10.92.111.189/stream", ... },
}
```

Stream settings in `config.py` match the sketch:
```python
CAMERA = {
    "frame_width":  320,    # FRAMESIZE_QVGA
    "frame_height": 240,    # FRAMESIZE_QVGA
    "fps":          15,     # STREAM_TARGET_FPS in sketch
}
```

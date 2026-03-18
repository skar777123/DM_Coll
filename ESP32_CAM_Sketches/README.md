# ESP32-CAM Static IP Configuration

## How to Set Static IPs on Your ESP32-CAMs

Each ESP32-CAM needs a **unique static IP** so the Raspberry Pi can always find it,
even when the WiFi network changes or the router reboots.

## Step-by-Step Instructions

### 1. Open the CameraWebServer sketch in Arduino IDE

### 2. Add the following code BEFORE `WiFi.begin()` in the sketch:

```cpp
// ═══════════════════════════════════════════════════════════
//  STATIC IP CONFIGURATION — Change for each ESP32-CAM!
// ═══════════════════════════════════════════════════════════

// LEFT CAMERA: 192.168.1.181
// RIGHT CAMERA: 192.168.1.182  
// REAR CAMERA: 192.168.1.183

// Set YOUR values below:
IPAddress local_IP(192, 168, 1, 181);      // ← Change per camera
IPAddress gateway(192, 168, 1, 1);         // ← Your router IP
IPAddress subnet(255, 255, 255, 0);
IPAddress dns(8, 8, 8, 8);                 // Google DNS

// Apply static IP BEFORE WiFi.begin()
if (!WiFi.config(local_IP, gateway, subnet, dns)) {
  Serial.println("Static IP configuration failed!");
}
```

### 3. Flash each ESP32-CAM with a different IP:
- **Left camera**:  `192.168.1.181`
- **Right camera**: `192.168.1.182`
- **Rear camera**:  `192.168.1.183`

### 4. Update `config.py` on the Raspberry Pi:
```python
CAMERA_PORTS = {
    "left":  { "url": "http://192.168.1.181/stream", ... },
    "right": { "url": "http://192.168.1.182/stream", ... },
    "rear":  { "url": "http://192.168.1.183/stream", ... },
}
```

### 5. Finding Your Gateway IP
- On Windows: `ipconfig` → look for "Default Gateway"
- On Linux/Pi: `ip route | grep default`
- Usually it's `192.168.1.1` or `192.168.0.1`

## How to Find the Right Subnet
Your static IPs must be in the **same subnet** as your router.
If your router is `192.168.1.1`, use `192.168.1.xxx`.
If your router is `10.132.20.1`, use `10.132.20.xxx`.

Pick numbers above 200 to avoid conflicts with DHCP range.

## Alternative: mDNS (Zero-Config Networking)
If you can't set static IPs, flash each ESP32 with a unique mDNS hostname.
The scanner.py on the Pi will automatically find them by name.
See `esp32_cam_mdns.ino` for the mDNS-enabled sketch.

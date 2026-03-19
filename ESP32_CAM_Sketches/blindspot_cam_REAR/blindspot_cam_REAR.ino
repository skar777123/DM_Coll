/*
 * ═══════════════════════════════════════════════════════════════
 *  blindspot_cam_REAR.ino
 *  BlindSpotGuard — REAR Camera (10.92.111.189)
 *
 *  Flash this to the REAR ESP32-CAM.
 *
 *  Endpoints served (port 80):
 *    /stream   → MJPEG feed → sensors/camera.py CameraStream
 *    /capture  → single JPEG
 *    /id       → {"position":"rear","ip":"...","hostname":"..."}
 *    /health   → {"status":"ok","uptime_ms":...,"rssi":...}
 *
 *  Auto-discovery: scanner.py runs nmap then calls /id to assign
 *  "rear" to stream URL http://10.92.111.189/stream
 * ═══════════════════════════════════════════════════════════════
 */

// ── Camera identity — do NOT change for this file ────────────────────────────
const char* CAMERA_POSITION = "rear";
const char* MDNS_HOSTNAME   = "blindspot-rear";   // resolves as blindspot-rear.local

// ── Static IP — must match config.py CAMERA_PORTS["rear"]["url"] ─────────────
// config.py: "url": "http://10.92.111.189/stream"
IPAddress STATIC_IP  (10, 92, 111, 189);   // ← Unique to REAR camera
IPAddress GATEWAY_IP (10, 92, 111,   156);   // ← Your router IP
IPAddress SUBNET_MASK(255, 255, 255,  0);
IPAddress DNS_IP     (  8,   8,   8,  8);

// Must include AFTER the identity/IP variables (header uses extern references)
#include "blindspot_config.h"


void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(false);
    delay(100);

    Serial.println("\n═══════════════════════════════════════════════");
    Serial.printf( "  BlindSpotGuard  |  Camera: %s\n", CAMERA_POSITION);
    Serial.printf( "  Static IP: %s\n", STATIC_IP.toString().c_str());
    Serial.println("═══════════════════════════════════════════════");

    // 1. Initialise camera hardware
    if (!initCamera()) {
        Serial.println("[SETUP] Camera init failed — restarting in 5 s");
        delay(5000);
        ESP.restart();
    }

    // 2. Connect to WiFi with static IP
    connectWiFi();

    // 3. Register mDNS hostname (blindspot-rear.local)
    setupMDNS();

    // 4. Start HTTP server on port 80
    startCameraServer();

    // ── Print access URLs ─────────────────────────────────────────────────────
    Serial.println("═══════════════════════════════════════════════");
    Serial.printf( "  MJPEG stream : http://%s/stream\n",
                   WiFi.localIP().toString().c_str());
    Serial.printf( "  Single frame : http://%s/capture\n",
                   WiFi.localIP().toString().c_str());
    Serial.printf( "  Identity JSON: http://%s/id\n",
                   WiFi.localIP().toString().c_str());
    Serial.printf( "  mDNS URL     : http://blindspot-rear.local/stream\n");
    Serial.println("═══════════════════════════════════════════════");
    Serial.println("  Ready — streaming to Raspberry Pi");
    Serial.println("═══════════════════════════════════════════════\n");
}


void loop() {
    // WiFi watchdog — auto-reconnects and restarts if WiFi fails
    wifiWatchdog();

    // Periodic status log every 30 s
    static unsigned long lastStatus = 0;
    if (millis() - lastStatus > 30000) {
        lastStatus = millis();
        Serial.printf("[%s] Uptime: %lu s | IP: %s | RSSI: %d dBm\n",
                      CAMERA_POSITION,
                      millis() / 1000,
                      WiFi.localIP().toString().c_str(),
                      WiFi.RSSI());
    }

    delay(5000);
}

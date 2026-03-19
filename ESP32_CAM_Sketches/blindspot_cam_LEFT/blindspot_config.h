/*
 * ═══════════════════════════════════════════════════════════════
 *  blindspot_config.h — Shared Configuration for all three cams
 *  Project: BlindSpotGuard — Real Vehicle Detection System
 * ═══════════════════════════════════════════════════════════════
 *
 *  This file is #included by each camera's .ino sketch.
 *  Only one camera is compiled and flashed at a time.
 *  Use the CAMERA_LEFT / CAMERA_RIGHT / CAMERA_REAR defines
 *  at the top of each .ino to select which one to build.
 *
 *  Matches config.py on the Raspberry Pi:
 *    CAMERA_PORTS["left"]  → 10.92.111.188
 *    CAMERA_PORTS["right"] → 10.92.111.190
 *    CAMERA_PORTS["rear"]  → 10.92.111.189
 *
 *  Matches sensors/scanner.py:
 *    Auto-discovery: nmap scans subnet → queries /id on each device
 *    /id  endpoint returns { "position": "left"|"right"|"rear", "ip": "...", "hostname": "..." }
 *
 *  Matches sensors/camera.py:
 *    /stream   → MJPEG multipart/x-mixed-replace  (primary, always active)
 *    /capture  → single JPEG                       (one-shot polling fallback)
 * ═══════════════════════════════════════════════════════════════
 */

#pragma once

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_http_server.h"

// ─────────────────────────────────────────────────────────────────────────────
//  CHANGE THESE FOR YOUR NETWORK
// ─────────────────────────────────────────────────────────────────────────────

#define WIFI_SSID     "Rohit Dakare"      // ← Your WiFi SSID
#define WIFI_PASS     "11111111"  // ← Your WiFi password

// Raspberry Pi IP — used only for reference / future OTA
// (Not strictly needed for streaming, but useful for logging)
#define RASPI_IP      "192.168.1.100"

// ─────────────────────────────────────────────────────────────────────────────
//  CAMERA PIN CONFIG — AI-Thinker ESP32-CAM (do NOT change)
// ─────────────────────────────────────────────────────────────────────────────

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_GPIO_NUM       4  // Flash LED (turn off to avoid glare)

// ─────────────────────────────────────────────────────────────────────────────
//  STREAM SETTINGS — must match sensors/camera.py CAMERA config
//  frame_width=320, frame_height=240, fps=15
// ─────────────────────────────────────────────────────────────────────────────

#define STREAM_FRAME_SIZE    FRAMESIZE_QVGA   // 320×240 matches config.py
#define STREAM_JPEG_QUALITY  12               // 0-63: lower = better, 12 = good balance
#define STREAM_TARGET_FPS    15               // matches CAMERA["fps"] in config.py
#define STREAM_FRAME_DELAY   (1000 / STREAM_TARGET_FPS)   // ms between frames

// ─────────────────────────────────────────────────────────────────────────────
//  MJPEG boundary string — must produce valid multipart/x-mixed-replace
//  that sensors/camera.py can parse (looking for \xff\xd8 SOI and \xff\xd9 EOI)
// ─────────────────────────────────────────────────────────────────────────────

#define PART_BOUNDARY "BlindSpotGuardBoundary"
static const char* _STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART         =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ─────────────────────────────────────────────────────────────────────────────
//  HTTP Server handles — one server on port 80 serving all endpoints
// ─────────────────────────────────────────────────────────────────────────────

httpd_handle_t _camera_httpd = NULL;

// ─────────────────────────────────────────────────────────────────────────────
//  Forward declarations (defined in each .ino using these)
// ─────────────────────────────────────────────────────────────────────────────

extern const char* CAMERA_POSITION;
extern const char* MDNS_HOSTNAME;
extern IPAddress   STATIC_IP;
extern IPAddress   GATEWAY_IP;
extern IPAddress   SUBNET_MASK;
extern IPAddress   DNS_IP;

// ─────────────────────────────────────────────────────────────────────────────
//  /stream handler — MJPEG multipart/x-mixed-replace
//  sensors/camera.py reads this via CameraStream._url_loop()
// ─────────────────────────────────────────────────────────────────────────────

static esp_err_t stream_handler(httpd_req_t* req) {
    camera_fb_t* fb = NULL;
    esp_err_t    res;
    char         part_buf[64];
    size_t       hlen;
    uint32_t     frame_count = 0;

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    // CORS — allow Raspberry Pi dashboard to fetch frames cross-origin
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control",               "no-cache");

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[STREAM] Camera frame capture failed — skipping");
            vTaskDelay(pdMS_TO_TICKS(100));   // brief pause before retry
            continue;
        }

        if (fb->format != PIXFORMAT_JPEG) {
            Serial.println("[STREAM] Warning: frame is not JPEG format");
            esp_camera_fb_return(fb);
            continue;
        }

        // Send multipart boundary
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }

        // Send MIME headers for this part
        hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len);
        res  = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }

        // Send the JPEG frame bytes
        // sensors/camera.py locates \xff\xd8 SOI and \xff\xd9 EOI to extract frames
        res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
        esp_camera_fb_return(fb);  // ALWAYS return the frame buffer
        fb = NULL;

        if (res != ESP_OK) break;

        frame_count++;
        if (frame_count % 150 == 0) {   // log every ~10 seconds at 15 FPS
            Serial.printf("[STREAM] %u frames sent | RSSI: %d dBm\n",
                          frame_count, WiFi.RSSI());
        }

        // Throttle to TARGET_FPS so we don't flood the Pi's socket buffer
        vTaskDelay(pdMS_TO_TICKS(STREAM_FRAME_DELAY));
    }

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
//  /capture handler — single JPEG (used by camera.py polling fallback)
//  Called by CameraStream._url_loop() when Content-Type is image/jpeg
// ─────────────────────────────────────────────────────────────────────────────

static esp_err_t capture_handler(httpd_req_t* req) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req,      "image/jpeg");
    httpd_resp_set_hdr(req,       "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req,       "Content-Disposition",         "inline");
    esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
//  /id handler — identity endpoint parsed by sensors/scanner.py
//
//  scanner.py calls _query_camera_id(ip) and expects JSON with:
//    { "position": "left"|"right"|"rear", "ip": "...", "hostname": "..." }
//
//  This response is what allows nmap auto-discovery to work correctly.
// ─────────────────────────────────────────────────────────────────────────────

static esp_err_t id_handler(httpd_req_t* req) {
    char response[192];
    snprintf(response, sizeof(response),
             "{\"position\":\"%s\",\"ip\":\"%s\",\"hostname\":\"%s.local\","
             "\"stream_url\":\"http://%s/stream\","
             "\"capture_url\":\"http://%s/capture\","
             "\"rssi\":%d}",
             CAMERA_POSITION,
             WiFi.localIP().toString().c_str(),
             MDNS_HOSTNAME,
             WiFi.localIP().toString().c_str(),
             WiFi.localIP().toString().c_str(),
             WiFi.RSSI());

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req,  "Access-Control-Allow-Origin", "*");
    return httpd_resp_sendstr(req, response);
}

// ─────────────────────────────────────────────────────────────────────────────
//  /health handler — quick liveness check for Raspberry Pi health monitor
//  Matches dashboard/app.py GET /api/health
// ─────────────────────────────────────────────────────────────────────────────

static esp_err_t health_handler(httpd_req_t* req) {
    char response[128];
    snprintf(response, sizeof(response),
             "{\"status\":\"ok\",\"position\":\"%s\",\"uptime_ms\":%lu,\"rssi\":%d}",
             CAMERA_POSITION, millis(), WiFi.RSSI());

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req,  "Access-Control-Allow-Origin", "*");
    return httpd_resp_sendstr(req, response);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Start HTTP server — registers all four endpoints on port 80
// ─────────────────────────────────────────────────────────────────────────────

void startCameraServer() {
    httpd_config_t srv_cfg = HTTPD_DEFAULT_CONFIG();
    srv_cfg.server_port         = 80;
    srv_cfg.ctrl_port           = 32768;
    srv_cfg.max_uri_handlers    = 8;
    srv_cfg.max_open_sockets    = 4;
    srv_cfg.recv_wait_timeout   = 10;
    srv_cfg.send_wait_timeout   = 10;
    srv_cfg.stack_size          = 8192;

    // /stream — MJPEG feed consumed by camera.py CameraStream._url_loop()
    httpd_uri_t stream_uri  = { "/stream",  HTTP_GET, stream_handler,  NULL };
    // /capture — single JPEG polled by camera.py as polling fallback
    httpd_uri_t capture_uri = { "/capture", HTTP_GET, capture_handler, NULL };
    // /id — parsed by scanner.py _query_camera_id() for auto-discovery
    httpd_uri_t id_uri      = { "/id",      HTTP_GET, id_handler,      NULL };
    // /health — liveness ping from Raspberry Pi
    httpd_uri_t health_uri  = { "/health",  HTTP_GET, health_handler,  NULL };

    if (httpd_start(&_camera_httpd, &srv_cfg) == ESP_OK) {
        httpd_register_uri_handler(_camera_httpd, &stream_uri);
        httpd_register_uri_handler(_camera_httpd, &capture_uri);
        httpd_register_uri_handler(_camera_httpd, &id_uri);
        httpd_register_uri_handler(_camera_httpd, &health_uri);
        Serial.println("[HTTP] Server started on port 80");
        Serial.printf( "[HTTP]   /stream   → MJPEG feed (%d FPS, QVGA)\n", STREAM_TARGET_FPS);
        Serial.printf( "[HTTP]   /capture  → single JPEG\n");
        Serial.printf( "[HTTP]   /id       → identity JSON {position, ip, hostname}\n");
        Serial.printf( "[HTTP]   /health   → liveness JSON\n");
    } else {
        Serial.println("[HTTP] FAILED to start server!");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Camera hardware initialisation (common setup)
// ─────────────────────────────────────────────────────────────────────────────

bool initCamera() {
    // Turn off flash LED — prevents glare on vehicle detection
    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);

    camera_config_t cam_cfg;
    cam_cfg.ledc_channel   = LEDC_CHANNEL_0;
    cam_cfg.ledc_timer     = LEDC_TIMER_0;
    cam_cfg.pin_d0         = Y2_GPIO_NUM;
    cam_cfg.pin_d1         = Y3_GPIO_NUM;
    cam_cfg.pin_d2         = Y4_GPIO_NUM;
    cam_cfg.pin_d3         = Y5_GPIO_NUM;
    cam_cfg.pin_d4         = Y6_GPIO_NUM;
    cam_cfg.pin_d5         = Y7_GPIO_NUM;
    cam_cfg.pin_d6         = Y8_GPIO_NUM;
    cam_cfg.pin_d7         = Y9_GPIO_NUM;
    cam_cfg.pin_xclk       = XCLK_GPIO_NUM;
    cam_cfg.pin_pclk       = PCLK_GPIO_NUM;
    cam_cfg.pin_vsync      = VSYNC_GPIO_NUM;
    cam_cfg.pin_href       = HREF_GPIO_NUM;
    cam_cfg.pin_sccb_sda   = SIOD_GPIO_NUM;
    cam_cfg.pin_sccb_scl   = SIOC_GPIO_NUM;
    cam_cfg.pin_pwdn       = PWDN_GPIO_NUM;
    cam_cfg.pin_reset      = RESET_GPIO_NUM;
    cam_cfg.xclk_freq_hz   = 20000000;       // 20 MHz XCLK
    cam_cfg.pixel_format   = PIXFORMAT_JPEG; // JPEG output for streaming
    cam_cfg.grab_mode      = CAMERA_GRAB_LATEST;  // always give newest frame

    if (psramFound()) {
        // PSRAM available: higher quality, double-buffered
        Serial.println("[CAM] PSRAM found — high-quality mode");
        cam_cfg.frame_size    = STREAM_FRAME_SIZE;
        cam_cfg.jpeg_quality  = STREAM_JPEG_QUALITY;    // e.g. 12
        cam_cfg.fb_count      = 2;
        cam_cfg.fb_location   = CAMERA_FB_IN_PSRAM;
    } else {
        // No PSRAM: conservative settings to avoid OOM
        Serial.println("[CAM] No PSRAM — conservative mode");
        cam_cfg.frame_size    = FRAMESIZE_QVGA;
        cam_cfg.jpeg_quality  = 14;   // slightly lower quality
        cam_cfg.fb_count      = 1;
        cam_cfg.fb_location   = CAMERA_FB_IN_DRAM;
    }

    esp_err_t err = esp_camera_init(&cam_cfg);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init FAILED! Error: 0x%x\n", err);
        return false;
    }

    // Fine-tune sensor settings for outdoor / vehicle detection
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);       // -2 to 2 — 0 = neutral
        s->set_contrast(s, 1);         // -2 to 2 — slight boost for vehicle edges
        s->set_saturation(s, 0);       // -2 to 2 — 0 = neutral
        s->set_sharpness(s, 1);        // -2 to 2 — slight sharpening
        s->set_exposure_ctrl(s, 1);    // 1 = auto exposure
        s->set_awb_gain(s, 1);         // 1 = auto white balance
        s->set_gain_ctrl(s, 1);        // 1 = auto gain
        s->set_whitebal(s, 1);         // 1 = auto WB
        s->set_aec2(s, 1);             // 1 = DSP AEC for daylight
        s->set_ae_level(s, 0);         // AE level: 0 = target middle
        s->set_aec_value(s, 300);      // AEC value when manual
        s->set_agc_gain(s, 0);         // AGC gain: 0 = 0x
        s->set_gainceiling(s, (gainceiling_t)6); // 128x max gain
        s->set_denoise(s, 1);          // enable noise reduction
        s->set_hmirror(s, 0);          // horizontal mirror — adjust per mount
        s->set_vflip(s, 0);            // vertical flip — adjust per mount
    }

    Serial.printf("[CAM] Initialised — %dx%d @ %d FPS JPEG quality=%d\n",
                  320, 240, STREAM_TARGET_FPS, cam_cfg.jpeg_quality);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  WiFi setup with static IP (matches config.py IP addresses)
// ─────────────────────────────────────────────────────────────────────────────

bool connectWiFi() {
    Serial.printf("[WiFi] Connecting to '%s' with static IP %s …\n",
                  WIFI_SSID, STATIC_IP.toString().c_str());

    // Apply static IP BEFORE WiFi.begin() — required on ESP32 Arduino
    if (!WiFi.config(STATIC_IP, GATEWAY_IP, SUBNET_MASK, DNS_IP)) {
        Serial.println("[WiFi] WARNING: Static IP config failed — falling back to DHCP");
    }

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);    // Disable WiFi power saving — prevents stream latency spikes
    WiFi.setTxPower(WIFI_POWER_19_5dBm); // Max power for range in vehicle mounting
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    uint8_t attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Connection FAILED — restarting in 3 s");
        delay(3000);
        ESP.restart();
        return false;
    }

    Serial.printf("[WiFi] Connected!\n");
    Serial.printf("[WiFi]   IP:      %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[WiFi]   Gateway: %s\n", WiFi.gatewayIP().toString().c_str());
    Serial.printf("[WiFi]   Signal:  %d dBm\n", WiFi.RSSI());

    // Verify we got the expected static IP
    if (WiFi.localIP() != STATIC_IP) {
        Serial.printf("[WiFi] WARNING: Got %s but expected %s — DHCP may have overridden\n",
                      WiFi.localIP().toString().c_str(), STATIC_IP.toString().c_str());
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  mDNS setup — allows scanner.py to find cameras by hostname
//  scanner.py also uses nmap + /id, so mDNS is a complementary discovery method
// ─────────────────────────────────────────────────────────────────────────────

void setupMDNS() {
    if (MDNS.begin(MDNS_HOSTNAME)) {
        MDNS.addService("http", "tcp", 80);
        MDNS.addServiceTxt("http", "tcp", "position", CAMERA_POSITION);
        MDNS.addServiceTxt("http", "tcp", "stream",   "/stream");
        MDNS.addServiceTxt("http", "tcp", "project",  "BlindSpotGuard");
        Serial.printf("[mDNS] Registered as http://%s.local\n", MDNS_HOSTNAME);
    } else {
        Serial.println("[mDNS] FAILED — IP-only discovery will still work via nmap");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  WiFi watchdog — called from loop() to auto-reconnect on drop
// ─────────────────────────────────────────────────────────────────────────────

void wifiWatchdog() {
    if (WiFi.status() == WL_CONNECTED) return;

    Serial.printf("[WiFi] Connection lost — reconnecting …\n");
    WiFi.reconnect();

    uint8_t retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 30) {
        delay(500);
        Serial.print(".");
        retries++;
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Reconnect failed — restarting ESP32");
        ESP.restart();
    } else {
        Serial.printf("[WiFi] Reconnected! IP: %s\n", WiFi.localIP().toString().c_str());
    }
}

/*
 * ═══════════════════════════════════════════════════════════════
 *  ESP32-CAM — LEFT Camera (BlindSpotGuard)
 *  Static IP: 192.168.1.181
 *  mDNS Hostname: blindspot-left.local
 * ═══════════════════════════════════════════════════════════════
 * 
 *  INSTRUCTIONS:
 *  1. Change WIFI_SSID and WIFI_PASS to your network
 *  2. Change STATIC_IP to match your subnet
 *  3. Change GATEWAY_IP to your router IP
 *  4. Flash to ESP32-CAM
 *  
 *  For RIGHT camera: Copy this file, change:
 *    - CAMERA_POSITION to "right"
 *    - STATIC_IP to {192, 168, 1, 182}
 *    - MDNS_HOSTNAME to "blindspot-right"
 *
 *  For REAR camera: Copy this file, change:
 *    - CAMERA_POSITION to "rear"
 *    - STATIC_IP to {192, 168, 1, 183}
 *    - MDNS_HOSTNAME to "blindspot-rear"
 * ═══════════════════════════════════════════════════════════════
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_http_server.h"

// ═══════════════ CHANGE THESE FOR YOUR SETUP ═══════════════

// WiFi credentials
const char* WIFI_SSID = "YOUR_WIFI_NAME";     // ← Your WiFi name
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";  // ← Your WiFi password

// Camera identity
const char* CAMERA_POSITION = "left";          // "left" | "right" | "rear"
const char* MDNS_HOSTNAME   = "blindspot-left"; // mDNS name (no .local suffix)

// Static IP configuration (CHANGE TO MATCH YOUR NETWORK!)
IPAddress STATIC_IP(192, 168, 1, 181);         // ← Unique per camera
IPAddress GATEWAY_IP(192, 168, 1, 1);          // ← Your router IP
IPAddress SUBNET_MASK(255, 255, 255, 0);
IPAddress DNS_IP(8, 8, 8, 8);

// ═══════════════ CAMERA PIN CONFIG (AI-Thinker) ═══════════════

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

#define LED_GPIO_NUM       4   // Flash LED

// ═══════════════ MJPEG STREAM HANDLER ═══════════════

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// Stream handler — serves MJPEG at /stream
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[64];
    
    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;
    
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    
    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            res = ESP_FAIL;
            break;
        }
        
        size_t hlen = snprintf(part_buf, 64, _STREAM_PART, fb->len);
        
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }
        
        res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }
        
        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }
        
        esp_camera_fb_return(fb);
        fb = NULL;
        
        delay(33); // ~30 FPS
    }
    
    return res;
}

// Identity handler — serves camera position at /id
static esp_err_t id_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    
    char response[128];
    snprintf(response, sizeof(response), 
             "{\"position\":\"%s\",\"ip\":\"%s\",\"hostname\":\"%s.local\"}",
             CAMERA_POSITION, WiFi.localIP().toString().c_str(), MDNS_HOSTNAME);
    
    return httpd_resp_sendstr(req, response);
}

// Single capture handler — serves single JPEG at /capture
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline");
    
    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}

void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    
    // Stream server on port 80
    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };
    
    httpd_uri_t id_uri = {
        .uri       = "/id",
        .method    = HTTP_GET,
        .handler   = id_handler,
        .user_ctx  = NULL
    };
    
    httpd_uri_t capture_uri = {
        .uri       = "/capture",
        .method    = HTTP_GET,
        .handler   = capture_handler,
        .user_ctx  = NULL
    };
    
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &stream_uri);
        httpd_register_uri_handler(camera_httpd, &id_uri);
        httpd_register_uri_handler(camera_httpd, &capture_uri);
        Serial.println("Camera HTTP server started on port 80");
    }
}

// ═══════════════ SETUP ═══════════════

void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();
    Serial.println("═══════════════════════════════════════");
    Serial.printf("  BlindSpotGuard ESP32-CAM [%s]\n", CAMERA_POSITION);
    Serial.println("═══════════════════════════════════════");
    
    // ── Camera Init ──
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_LATEST;
    
    // Resolution: QVGA (320x240) for best streaming performance
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;      // 0-63, lower = higher quality
    config.fb_count = 2;           // Double-buffered for smooth streaming
    config.fb_location = CAMERA_FB_IN_PSRAM;
    
    // Check for PSRAM
    if (psramFound()) {
        Serial.println("PSRAM found — using high quality settings");
        config.jpeg_quality = 10;
        config.fb_count = 2;
    } else {
        Serial.println("No PSRAM — using conservative settings");
        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 14;
        config.fb_count = 1;
    }
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed! Error: 0x%x\n", err);
        Serial.println("Restarting in 5 seconds...");
        delay(5000);
        ESP.restart();
    }
    Serial.println("Camera initialized successfully");
    
    // ── WiFi with Static IP ──
    Serial.printf("Connecting to WiFi: %s\n", WIFI_SSID);
    
    // Set static IP BEFORE WiFi.begin()
    if (!WiFi.config(STATIC_IP, GATEWAY_IP, SUBNET_MASK, DNS_IP)) {
        Serial.println("WARNING: Static IP config failed! Falling back to DHCP.");
    } else {
        Serial.printf("Static IP configured: %s\n", STATIC_IP.toString().c_str());
    }
    
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    WiFi.setSleep(false);  // Disable WiFi power saving for lower latency
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\nWiFi connection failed! Restarting...");
        delay(3000);
        ESP.restart();
    }
    
    Serial.println("\nWiFi Connected!");
    Serial.printf("  IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Gateway:    %s\n", WiFi.gatewayIP().toString().c_str());
    Serial.printf("  Signal:     %d dBm\n", WiFi.RSSI());
    
    // ── mDNS Setup ──
    if (MDNS.begin(MDNS_HOSTNAME)) {
        MDNS.addService("http", "tcp", 80);
        MDNS.addServiceTxt("http", "tcp", "position", CAMERA_POSITION);
        Serial.printf("  mDNS:       http://%s.local\n", MDNS_HOSTNAME);
    } else {
        Serial.println("  mDNS:       FAILED (will use IP only)");
    }
    
    // ── Start HTTP Server ──
    startCameraServer();
    
    Serial.println("═══════════════════════════════════════");
    Serial.printf("  Stream URL: http://%s/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("  ID URL:     http://%s/id\n", WiFi.localIP().toString().c_str());
    Serial.printf("  mDNS URL:   http://%s.local/stream\n", MDNS_HOSTNAME);
    Serial.println("═══════════════════════════════════════");
}

void loop() {
    // Watchdog: restart if WiFi drops
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi lost! Reconnecting...");
        WiFi.reconnect();
        int retries = 0;
        while (WiFi.status() != WL_CONNECTED && retries < 20) {
            delay(500);
            Serial.print(".");
            retries++;
        }
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("\nReconnect failed. Restarting ESP32...");
            ESP.restart();
        }
        Serial.println("\nReconnected!");
    }
    
    delay(10000);  // Check every 10 seconds
}

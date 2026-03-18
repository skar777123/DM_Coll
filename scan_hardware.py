"""
scan_hardware.py
────────────────
Diagnostic tool to verify hardware connectivity for BlindSpotGuard.
Tests camera streams via HTTP probing and ultrasonic GPIO sensors.

Usage:
    python3 scan_hardware.py
"""

import sys
import time

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def scan_cameras():
    print_header("1. Scanning for ESP32-CAMs (Wi-Fi)")
    try:
        from sensors.scanner import discover_esp32_cameras
        from config import CAMERA_PORTS
        
        print("Phase 1: Probing configured camera IPs...")
        for pos, cfg in CAMERA_PORTS.items():
            url = cfg.get("url", "N/A")
            print(f"  {pos.upper():5s} → {url}")
        
        print("\nStarting discovery scan...\n")
        urls = discover_esp32_cameras()
        
        if not urls:
            print("\n[FAIL] No active ESP32-CAM streams found on the network.")
            print("   Troubleshooting:")
            print("   - Are the cameras powered on?")
            print("   - Are they connected to the exact same Wi-Fi network?")
            print("   - Check that ESP32 is running CameraWebServer sketch")
            print("   - Try accessing http://<camera-ip>/stream in a browser")
            print("   - Is 'nmap' installed? (sudo apt install nmap)")
        else:
            print(f"\n[OK] Found {len(urls)} Active Camera Stream(s):")
            for pos, url in urls.items():
                print(f"   -> {pos.upper():5s} Camera -> {url}")
                
            # Quick frame test — try to grab one frame
            print("\nTesting frame capture...")
            import requests
            for pos, url in urls.items():
                try:
                    res = requests.get(url, stream=True, timeout=3)
                    ctype = res.headers.get("Content-Type", "")
                    res.close()
                    if "multipart" in ctype:
                        print(f"   [OK] {pos.upper():5s} — MJPEG stream confirmed")
                    elif "jpeg" in ctype:
                        print(f"   [OK] {pos.upper():5s} — JPEG snapshot confirmed")
                    else:
                        print(f"   [??] {pos.upper():5s} — Content-Type: {ctype}")
                except Exception as e:
                    print(f"   [FAIL] {pos.upper():5s} — {e}")
                
    except Exception as e:
        print(f"[FAIL] Error during camera scan: {e}")

def scan_ultrasonics():
    print_header("2. Testing Ultrasonic Sensors (GPIO)")
    try:
        import RPi.GPIO as GPIO
        print("RPi.GPIO library found. Attempting hardware test...")
        from config import ULTRASONIC_PINS
        from sensors.ultrasonic import UltrasonicSensor
        
        for name, config in ULTRASONIC_PINS.items():
            print(f"\nTesting {name.upper()} Sensor (TRIG: BCM {config['trig']}, ECHO: BCM {config['echo']})...")
            sensor = UltrasonicSensor(name, config['trig'], config['echo'])
            sensor.start()
            
            # Wait enough time for median filter to fill (5 readings × 60ms = 300ms min)
            time.sleep(0.5)
            
            dist = sensor.distance_cm
            zone = sensor.zone
            sensor.stop()
            
            if dist < 0:
                print(f"   [FAIL] {name.upper()}: OFFLINE (No response)")
                print(f"          → Check wiring: TRIG->BCM{config['trig']}, ECHO->BCM{config['echo']}")
                print(f"          → Verify voltage divider on ECHO (1kΩ + 2kΩ)")
                print(f"          → Ensure sensor has 5V power supply")
            elif dist < 2:
                print(f"   [WARN] {name.upper()}: {dist} cm — Too close (HC-SR04 min = 2cm)")
            else:
                print(f"   [OK] {name.upper()}: {dist:.1f} cm [{zone.upper()}]")
                
    except ImportError:
        print("[FAIL] Cannot test Ultrasonic Sensors.")
        print("   Reason: 'RPi.GPIO' is not installed or you are running on a non-Pi machine.")
        print("   This test must be run on the actual Raspberry Pi.")
    except Exception as e:
        print(f"[FAIL] Error during ultrasonic test: {e}")

def scan_system():
    print_header("3. System Configuration Check")
    from config import ULTRASONIC, CAMERA, ZONE
    
    print(f"  Ultrasonic polling:  {ULTRASONIC['polling_interval_s']*1000:.0f} ms ({1/ULTRASONIC['polling_interval_s']:.0f} Hz)")
    print(f"  Median filter size:  {ULTRASONIC['readings_per_avg']} readings")
    print(f"  Max distance:        {ULTRASONIC['max_distance_cm']} cm")
    print(f"  Timeout:             {ULTRASONIC['timeout_s']*1000:.0f} ms")
    print(f"  Speed of sound:      {ULTRASONIC['speed_of_sound_cmps']} cm/s")
    print()
    print(f"  Zone thresholds:")
    print(f"    Safe:     > {ZONE['safe']} cm")
    print(f"    Caution:  ≤ {ZONE['caution']} cm")
    print(f"    Critical: ≤ {ZONE['critical']} cm")
    print()
    print(f"  Camera FPS:         {CAMERA['fps']}")
    print(f"  YOLO model:         {CAMERA['yolo_model']}")
    print(f"  Stream timeout:     {CAMERA.get('stream_frame_timeout', 'N/A')}s")
    print(f"  Stream chunk size:  {CAMERA.get('stream_chunk_size', 'N/A')} bytes")

if __name__ == "__main__":
    print_header("BLINDSPOT GUARD — HARDWARE DIAGNOSTICS")
    scan_system()
    scan_cameras()
    scan_ultrasonics()
    print("\n" + "="*60)
    print(" Diagnostics Complete.")
    print("="*60 + "\n")

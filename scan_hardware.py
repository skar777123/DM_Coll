"""
scan_hardware.py
────────────────
Diagnostic tool to verify hardware connectivity for BlindSpotGuard.
Runs the nmap IP scanner for ESP32-CAMs and tests Ultrasonic GPIO pins.
"""

import sys
import time

def print_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def scan_cameras():
    print_header("1. Scanning for ESP32-CAMs (Wi-Fi)")
    try:
        from sensors.scanner import discover_esp32_cameras
        print("Starting nmap scan on local subnet for active HTTP streams...\n")
        urls = discover_esp32_cameras()
        
        if not urls:
            print("\n[FAIL] No active ESP32-CAM streams found on the network.")
            print("   Troubleshooting:")
            print("   - Are the cameras powered on?")
            print("   - Are they connected to the exact same Wi-Fi network as this computer?")
            print("   - Is 'nmap' installed on this machine?")
        else:
            print("\n[OK] Found Active Camera Streams:")
            for pos, url in urls.items():
                print(f"   -> {pos.upper():5s} Camera -> {url}")
                
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
            
            # Wait 1s to let it gather smoothed readings
            time.sleep(1.0)
            
            dist = sensor.distance_cm
            sensor.stop()
            
            if dist < 0:
                print(f"   [FAIL] {name.upper()}: OFFLINE (No response - Check Wiring or Resistor Divider!)")
            else:
                print(f"   [OK] {name.upper()}: ONLINE (Current Distance: {dist} cm)")
                
    except ImportError:
        print("[FAIL] Cannot test Ultrasonic Sensors.")
        print("   Reason: 'RPi.GPIO' is not installed or you are running this on a Windows PC.")
        print("   This test must be run on the actual Raspberry Pi.")
    except Exception as e:
        print(f"[FAIL] Error during ultrasonic test: {e}")

if __name__ == "__main__":
    print_header("BLINDSPOT GUARD — HARDWARE DIAGNOSTICS")
    scan_cameras()
    scan_ultrasonics()
    print("\n" + "="*50)
    print(" Diagnostics Complete.")
    print("="*50 + "\n")

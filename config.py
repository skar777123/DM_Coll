"""
==============================================================
  Blind Spot Detection System — Configuration
  Project: Bike/Vehicle Blind Spot Warning System
  Hardware: Raspberry Pi 4B + ESP32-CAM x3 + HC-SR04 x3
==============================================================
"""

# ─────────────────────────────────────────────────────────────
#  GPIO PIN CONFIGURATION  (BCM numbering)
# ─────────────────────────────────────────────────────────────

ULTRASONIC_PINS = {
    "left": {
        "trig": 23,   # BCM 23 → Physical Pin 16
        "echo": 24,   # BCM 24 → Physical Pin 18
        "label": "Left Blindspot",
    },
    "right": {
        "trig": 17,   # BCM 17 → Physical Pin 11
        "echo": 27,   # BCM 27 → Physical Pin 13
        "label": "Right Blindspot",
    },
    "rear": {
        "trig": 5,    # BCM 5  → Physical Pin 29
        "echo": 6,    # BCM 6  → Physical Pin 31
        "label": "Rear Sensor",
    },
}

# LED Output Pins (BCM)
LED_PINS = {
    "left":  19,   # Left dashboard LED
    "right": 26,   # Right dashboard LED
    "rear":  13,   # Rear/Center dashboard LED
}

# Vibration Motor Output Pins (BCM)
MOTOR_PINS = {
    "left":  20,   # Left seat/wheel motor
    "right": 21,   # Right seat/wheel motor
}

# ─────────────────────────────────────────────────────────────
#  ESP32-CAM SERIAL UART PORTS
# ─────────────────────────────────────────────────────────────

CAMERA_PORTS = {
    "left":  {
        "port":  "/dev/ttyAMA0",
        "url":   "http://10.132.20.188/stream",
        "baud":  115200,
        "label": "Left Camera",
    },
    "right": {
        "port":  "/dev/ttyAMA4",
        "url":   "http://10.132.20.101/stream",
        "baud":  115200,
        "label": "Right Camera",
    },
    "rear":  {
        "port":  "/dev/ttyAMA2",
        "url":   "http://10.132.20.209/stream",
        "baud":  115200,
        "label": "Rear Camera",
    },
}

# ─────────────────────────────────────────────────────────────
#  DETECTION ZONE THRESHOLDS (in centimetres)
# ─────────────────────────────────────────────────────────────

ZONE = {
    "safe":     300,   # > 300 cm  → Green
    "caution":  200,   # < 200 cm  → Yellow (if vehicle + moving)
    "critical": 100,   # < 100 cm  → Red    (if vehicle + moving)
}

# ─────────────────────────────────────────────────────────────
#  ULTRASONIC SENSOR SETTINGS
# ─────────────────────────────────────────────────────────────

ULTRASONIC = {
    "trigger_pulse_us":    0.00001,   # 10 µs trigger pulse
    "max_distance_cm":     400,       # Beyond this → ignored
    "speed_of_sound_cmps": 34300,     # cm/s at ~20°C
    "timeout_s":           0.03,      # 30 ms timeout (≈ 500 cm round-trip)
    "readings_per_avg":    5,         # Median filter window size (odd number)
    "polling_interval_s":  0.06,      # ~16 Hz polling rate (HC-SR04 min cycle = 60ms)
    "settle_time_s":       0.000005,  # 5 µs pre-trigger LOW settle time
    "outlier_jump_pct":    0.50,      # Reject readings that jump >50% from prev
    "min_valid_cm":        2.0,       # HC-SR04 min reliable range
}

# ─────────────────────────────────────────────────────────────
#  CAMERA / OBJECT DETECTION SETTINGS
# ─────────────────────────────────────────────────────────────

CAMERA = {
    "frame_width":  320,
    "frame_height": 240,
    "fps":          15,
    "yolo_model":   "yolov8n.pt",          # Nano model for Pi performance
    "conf_thresh":  0.45,
    "target_classes":  ["car", "truck", "motorcycle", "bus", "person", "bicycle"],
    "vehicle_classes": ["car", "truck", "motorcycle", "bus"],
    "approach_speed_thresh_px": 8,         # px/frame velocity to trigger caution
    "motion_thresh_px":         3,         # min px shift to consider "moving"
    "stream_chunk_size":     32768,        # 32 KB chunks for MJPEG reads
    "stream_frame_timeout":    5.0,        # Reconnect if no frame in 5 seconds
    "stream_connect_timeout":  5.0,        # TCP connect timeout for camera URLs
    "stream_read_timeout":    10.0,        # TCP read timeout for camera streams
    "stream_max_buffer":  1048576,         # 1 MB max buffer before reset
    "stream_retry_delay":      2.0,        # Initial retry delay (seconds)
    "stream_max_retry_delay": 10.0,        # Maximum retry delay (seconds)
}

# ─────────────────────────────────────────────────────────────
#  WARNING / FEEDBACK SETTINGS
# ─────────────────────────────────────────────────────────────

WARNING = {
    "led_flash_hz":          4,     # Flash rate in critical zone
    "motor_pulse_hz":        3,     # Vibration pulse rate in critical zone
    "motor_duty_caution":    0,     # Duty cycle % for caution (0 = off)
    "motor_duty_critical":   80,    # Duty cycle % for critical vibration
    "pwm_frequency_hz":      100,   # PWM carrier frequency for motors
}

# ─────────────────────────────────────────────────────────────
#  WEB DASHBOARD SETTINGS
# ─────────────────────────────────────────────────────────────

DASHBOARD = {
    "host":      "0.0.0.0",
    "port":      5000,
    "debug":     False,
    "emit_rate": 0.05,   # 20 Hz dashboard update
}

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────

LOGGING = {
    "level":   "INFO",
    "logfile": "logs/blind_spot.log",
    "max_mb":  10,
    "backups": 3,
}

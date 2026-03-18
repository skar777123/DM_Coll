"""
ML_Model/yolo/augment.py
─────────────────────────
Custom augmentation profile for blind-spot dashcam imagery.

Rationale
─────────
Dashcam / wing-mirror images share specific characteristics:
  • Objects often appear small, at the edges of frame (lane change scenarios)
  • Heavy motion blur at highway speeds
  • Extreme lighting variation (sunrise/sunset, tunnels, night)
  • Partial occlusion (other vehicles cutting in)
  • Rain / fog / glare distortion

Standard YOLO augmentation is tuned for general COCO images.
This profile emphasises the challenges above:
  • High HSV shift + brightness variation for lighting robustness
  • MixUp for handling occlusion
  • Heavy motion blur simulation (translate + shear)
  • Perspective warp for mirror-view geometry
  • Lower mosaic prob (dashcam images are already complex)
"""

# All keys are ultralytics Trainer hyperparameter names
# Reference: https://docs.ultralytics.com/usage/cfg/#augmentation-settings

AUGMENT_ARGS = {
    # ── Colour / Photometric ─────────────────────────────────────────────────
    "hsv_h":       0.020,    # hue shift       (±0.02 of 360°)
    "hsv_s":       0.80,     # saturation shift (±80%)
    "hsv_v":       0.50,     # value/brightness (±50%) — key for lighting

    # ── Geometric ────────────────────────────────────────────────────────────
    "degrees":     5.0,      # ± rotation (dashcam tilt)
    "translate":   0.15,     # ± x/y translation fraction (object at frame edge)
    "scale":       0.60,     # ± scale factor (small → large objects)
    "shear":       4.0,      # ± shear (motion blur simulation)
    "perspective": 0.0005,   # perspective warp (mirror/fisheye geometry)
    "flipud":      0.0,      # no vertical flip (dashcam is always upright)
    "fliplr":      0.50,     # horizontal flip (left/right symmetry in traffic)

    # ── Mosaic / Mixing ──────────────────────────────────────────────────────
    "mosaic":      0.80,     # mosaic probability (4-image tiles)
    "mixup":       0.15,     # mixup probability (handles occlusion cases)
    "copy_paste":  0.05,     # copy-paste augmentation
    "erasing":     0.40,     # random-erase rectangles (partial occlusion)

    # ── Other ────────────────────────────────────────────────────────────────
    "auto_augment": "randaugment",   # additional automatic augmentation policy
    "close_mosaic": 10,      # disable mosaic in last N epochs for stability
}

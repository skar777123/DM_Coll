"""
check_system.py
───────────────
Quick verification script for BlindSpotGuard ML setup.
"""

import os
import sys

def check():
    print("Checking BlindSpotGuard ML System...")
    
    paths = {
        "LSTM Model": "ML_Model/saved_models/threat_lstm.pt",
        "FusionNet": "ML_Model/saved_models/fusion_net.pt",
        "Scaler": "ML_Model/saved_models/scaler.pkl",
    }
    
    all_ok = True
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"  [OK] {name:12s} found at {path}")
        else:
            print(f"  [MISSING] {name:12s} NOT found at {path}")
            all_ok = False
            
    if all_ok:
        print("\nSUCCESS: ML models are ready for Two-Stage Verification.")
        print("Logic: 1. YOLO identifies vehicle -> 2. LSTM verifies movement/speed.")
    else:
        print("\nWARNING: Some models are missing. Run: python3 ML_Model/train.py")

if __name__ == "__main__":
    check()

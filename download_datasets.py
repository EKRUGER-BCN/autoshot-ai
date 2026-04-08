#!/usr/bin/env python3
"""
Download all datasets for AUTOSHOT retraining.
Includes: Roboflow CAPSTONE, GJ PW, and optional Kaggle datasets.
"""

import os
import sys
import subprocess
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"📦 Downloading datasets to {DATA_DIR}")
print("=" * 60)

# Roboflow API key from environment
ROBOFLOW_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_KEY:
    print("❌ ROBOFLOW_API_KEY not set. Exiting.")
    sys.exit(1)

# 1. Download Roboflow CAPSTONE
print("\n1️⃣  Downloading Roboflow CAPSTONE (~3,226 images)...")
capstone_code = os.system(f"""
python3 << 'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="{ROBOFLOW_KEY}")
proj = rf.workspace("capstone-nh0nc").project("car-damage-detection-t0g92")
proj.version(1).download("yolov8", location="{DATA_DIR}/roboflow_capstone", overwrite=True)
print("✅ CAPSTONE downloaded")
EOF
""")

# 2. Download Roboflow GJ PW
print("\n2️⃣  Downloading Roboflow GJ PW (~3,291 images)...")
gjpw_code = os.system(f"""
python3 << 'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="{ROBOFLOW_KEY}")
proj = rf.workspace("gj-pw-6vszg").project("car-damage-detection-nvhne")
proj.version(1).download("yolov8", location="{DATA_DIR}/roboflow_gjpw", overwrite=True)
print("✅ GJ PW downloaded")
EOF
""")

# 3. Optional: Kaggle datasets
print("\n3️⃣  Checking for Kaggle API credentials...")
kaggle_dir = Path.home() / ".kaggle" / "kaggle.json"
if kaggle_dir.exists():
    print("   Kaggle credentials found. Attempting download...")
    print("   ⚠️  Manual step: Install kaggle-api and download datasets")
    print("   Command: kaggle datasets download -d anujms/car-damage-detection")
else:
    print("   ⚠️  No Kaggle credentials found. Skipping Kaggle datasets.")
    print("   To enable: pip install kaggle && kaggle config set --quiet")

# Summary
print("\n" + "=" * 60)
print("📊 Download Summary:")
print(f"   CAPSTONE: {DATA_DIR}/roboflow_capstone")
print(f"   GJ PW:    {DATA_DIR}/roboflow_gjpw")
print(f"   Kaggle:   (Manual)")
print("\n✅ Download phase complete. Next: combine datasets and retrain.")

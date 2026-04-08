#!/usr/bin/env python3
"""
Build FINAL production dataset: 19,138 images
- CarDD (remapped): 1,598
- CAPSTONE: 3,595
- VehiDE (converted): 13,945
"""

import shutil
import random
from pathlib import Path
import yaml

BASE = Path("data/raw")
OUTPUT = Path("data/production_final")

CLASSES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass_shatter",
    4: "lamp_broken",
    5: "tire_flat",
    6: "paint_damage",
}

DATASETS = {
    "cardd": BASE / "cardd_remapped",
    "capstone": BASE / "roboflow_capstone",
    "vehide": BASE / "vehide_yolo",
}

def setup():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            (OUTPUT / split / subdir).mkdir(parents=True, exist_ok=True)

def process_dataset(name, source_dir):
    if not source_dir.exists():
        print(f"⚠️  {name} not found")
        return 0

    count = 0
    for split in ["train", "val", "test"]:
        img_dir = source_dir / split / "images"
        lbl_dir = source_dir / split / "labels"

        if not img_dir.exists():
            continue

        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in [".jpg", ".png"]:
                continue

            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue

            # Random split 70/15/15
            rand = random.random()
            out_split = "train" if rand < 0.70 else ("val" if rand < 0.85 else "test")

            dst_img = OUTPUT / out_split / "images" / f"{name}_{img.name}"
            dst_lbl = OUTPUT / out_split / "labels" / f"{name}_{img.stem}.txt"

            shutil.copy2(img, dst_img)
            shutil.copy2(lbl, dst_lbl)
            count += 1

    print(f"✅ {name:12} | {count:6} images")
    return count

def create_yaml():
    data = {
        "path": str(OUTPUT.absolute()),
        "train": str((OUTPUT / "train" / "images").absolute()),
        "val": str((OUTPUT / "val" / "images").absolute()),
        "test": str((OUTPUT / "test" / "images").absolute()),
        "nc": 7,
        "names": CLASSES,
    }
    with open(OUTPUT / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)

print("=" * 70)
print("🏗️  BUILDING FINAL PRODUCTION DATASET")
print("=" * 70)

setup()

total = 0
for name, path in DATASETS.items():
    total += process_dataset(name, path)

train = len(list((OUTPUT / "train" / "images").glob("*")))
val = len(list((OUTPUT / "val" / "images").glob("*")))
test = len(list((OUTPUT / "test" / "images").glob("*")))
total_split = train + val + test

create_yaml()

print("\n" + "=" * 70)
print("📊 FINAL DATASET — READY FOR TRAINING")
print("=" * 70)
print(f"Total images: {total_split:,}")
print(f"  Train: {train:,} ({train/total_split*100:.1f}%)")
print(f"  Val:   {val:,} ({val/total_split*100:.1f}%)")
print(f"  Test:  {test:,} ({test/total_split*100:.1f}%)")
print(f"\n🎯 Baseline: mAP50 = 0.614")
print(f"🚀 Target:  mAP50 ≥ 0.65+")
print(f"📁 Dataset: {OUTPUT}")
print(f"📋 Config:  {OUTPUT / 'data.yaml'}")
print("=" * 70)

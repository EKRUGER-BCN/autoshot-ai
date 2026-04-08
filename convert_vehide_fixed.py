#!/usr/bin/env python3
"""
Convert VehiDE (VIA polygon format) to YOLO bbox format — FIXED.
Handles nested directories and reads image dimensions from files.
"""

import json
import shutil
import random
from pathlib import Path
from PIL import Image

DATA_RAW = Path("data/raw")
VEHIDE_IMAGES = {
    "train": DATA_RAW / "image" / "image",
    "val": DATA_RAW / "validation" / "validation",
}
VEHIDE_ANNOS = {
    "train": DATA_RAW / "0Train_via_annos.json",
    "val": DATA_RAW / "0Val_via_annos.json",
}
VEHIDE_OUT = DATA_RAW / "vehide_yolo"

def setup_dirs():
    if VEHIDE_OUT.exists():
        shutil.rmtree(VEHIDE_OUT)
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            (VEHIDE_OUT / split / subdir).mkdir(parents=True, exist_ok=True)

def polygon_to_bbox(all_x, all_y, img_width, img_height):
    """Convert polygon to YOLO bbox."""
    if not all_x or not all_y or img_width == 0 or img_height == 0:
        return None

    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)

    center_x = (min_x + max_x) / 2 / img_width
    center_y = (min_y + max_y) / 2 / img_height
    width = (max_x - min_x) / img_width
    height = (max_y - min_y) / img_height

    # Clamp
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0.01, min(1, width))
    height = max(0.01, min(1, height))

    return center_x, center_y, width, height

def convert_dataset():
    print("=" * 70)
    print("🔄 CONVERTING VehiDE (Polygons → YOLO Bboxes) — FIXED")
    print("=" * 70)

    setup_dirs()

    converted = 0
    skipped = 0

    # Process both train and val annotations
    for split_name in ["train", "val"]:
        print(f"\n📋 Processing {split_name} set...")

        anno_file = VEHIDE_ANNOS[split_name]
        img_dir = VEHIDE_IMAGES[split_name]

        with open(anno_file) as f:
            annos = json.load(f)

        for img_name, img_data in annos.items():
            try:
                # Find image file
                img_path = img_dir / img_name
                if not img_path.exists():
                    skipped += 1
                    continue

                # Get image dimensions from file
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except:
                    skipped += 1
                    continue

                regions = img_data.get("regions", [])
                if not regions:
                    skipped += 1
                    continue

                # Create YOLO label
                label_lines = []

                for region in regions:
                    # Get class
                    class_name = region.get("class", "mat_bo_phan")
                    # Default mapping: everything to dent (0) for now
                    class_id = 0

                    # Get polygon coordinates (directly on region, not nested)
                    all_x = region.get("all_x", [])
                    all_y = region.get("all_y", [])

                    if not all_x or not all_y:
                        continue

                    bbox = polygon_to_bbox(all_x, all_y, img_width, img_height)
                    if not bbox:
                        continue

                    cx, cy, w, h = bbox
                    label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                if not label_lines:
                    skipped += 1
                    continue

                # Random split
                rand = random.random()
                if rand < 0.70:
                    out_split = "train"
                elif rand < 0.85:
                    out_split = "val"
                else:
                    out_split = "test"

                # Copy image
                shutil.copy2(img_path, VEHIDE_OUT / out_split / "images" / img_name)

                # Save label
                label_file = VEHIDE_OUT / out_split / "labels" / (Path(img_name).stem + ".txt")
                with open(label_file, "w") as f:
                    f.writelines(label_lines)

                converted += 1

                if converted % 2000 == 0:
                    print(f"   Converted {converted}...")

            except Exception as e:
                skipped += 1
                continue

    print("\n" + "=" * 70)
    print(f"✅ Converted: {converted} images")
    print(f"⚠️  Skipped: {skipped} images")
    print(f"📁 Output: {VEHIDE_OUT}")
    print("=" * 70)

if __name__ == "__main__":
    convert_dataset()

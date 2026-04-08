#!/usr/bin/env python3
"""
Remap CarDD 13-class labels to our 7-class system.
Recovers all 8k images instead of filtering.
"""

from pathlib import Path
import shutil

CARDD_SRC = Path("data/raw/cardd_roboflow")
CARDD_OUT = Path("data/raw/cardd_remapped")

# CarDD 13 classes → our 7 classes
CLASS_MAP = {
    0: 2,   # car-part-crack → crack
    1: 2,   # crack → crack
    2: 0,   # detachment → dent
    3: 5,   # flat-tire → tire_flat
    4: 3,   # glass-crack → glass_shatter
    5: 4,   # lamp-crack → lamp_broken
    6: 0,   # minor-deformation → dent
    7: 0,   # moderate-deformation → dent
    8: 6,   # paint-chips → paint_damage
    9: 1,   # scratch → scratch
    10: 1,  # scratches → scratch
    11: 0,  # severe-deformation → dent
    12: 2,  # side-mirror-crack → crack
}

def remap_labels():
    """Remap all CarDD labels."""
    if CARDD_OUT.exists():
        shutil.rmtree(CARDD_OUT)

    # Copy structure
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            (CARDD_OUT / split / subdir).mkdir(parents=True, exist_ok=True)

    total_remapped = 0
    total_skipped = 0

    for split in ["train", "val", "test"]:
        img_src = CARDD_SRC / split / "images"
        lbl_src = CARDD_SRC / split / "labels"
        img_dst = CARDD_OUT / split / "images"
        lbl_dst = CARDD_OUT / split / "labels"

        if not img_src.exists():
            continue

        for img in sorted(img_src.glob("*")):
            if img.suffix.lower() not in [".jpg", ".png"]:
                continue

            lbl = lbl_src / (img.stem + ".txt")
            if not lbl.exists():
                total_skipped += 1
                continue

            # Copy image
            shutil.copy2(img, img_dst / img.name)

            # Remap labels
            try:
                with open(lbl) as f:
                    lines = f.readlines()

                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    old_class = int(parts[0])
                    new_class = CLASS_MAP.get(old_class, -1)

                    if new_class == -1:
                        # Skip invalid class
                        continue

                    # Remap class ID
                    parts[0] = str(new_class)
                    remapped_lines.append(" ".join(parts) + "\n")

                # Write remapped labels
                if remapped_lines:
                    with open(lbl_dst / (img.stem + ".txt"), "w") as f:
                        f.writelines(remapped_lines)
                    total_remapped += 1
                else:
                    total_skipped += 1

            except Exception as e:
                total_skipped += 1
                continue

    return total_remapped, total_skipped

print("=" * 70)
print("🔄 REMAPPING CarDD LABELS (13 classes → 7 classes)")
print("=" * 70)
print("Mapping:")
for old_id, new_id in sorted(CLASS_MAP.items()):
    old_names = [
        "car-part-crack", "crack", "detachment", "flat-tire", "glass-crack",
        "lamp-crack", "minor-deformation", "moderate-deformation", "paint-chips",
        "scratch", "scratches", "severe-deformation", "side-mirror-crack"
    ]
    new_names = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat", "paint_damage"]
    print(f"  {old_id:2} ({old_names[old_id]:22}) → {new_id} ({new_names[new_id]})")

remapped, skipped = remap_labels()

print("\n" + "=" * 70)
print(f"✅ Remapped: {remapped} images")
print(f"⚠️  Skipped: {skipped} images")
print(f"📁 Output: {CARDD_OUT}")
print("=" * 70)

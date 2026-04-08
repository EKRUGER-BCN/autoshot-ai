#!/usr/bin/env python3
"""
PRODUCTION TRAINING: YOLOv8-XL on 19,138 images
Target: Beat v5_final (mAP50: 0.614, Precision: 0.684, Recall: 0.593)
"""

import json
import torch
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# Config
DATA_YAML = "data/production_final/data.yaml"
MODEL_SIZE = "xl"  # Largest model for best accuracy
EPOCHS = 150
IMG_SIZE = 640
BATCH_SIZE = 32  # Larger batch on GPU
PATIENCE = 20  # Early stopping
DEVICE = 0 if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("runs/detect")
PROJECT_NAME = "autoshot_v6_final"

print("=" * 80)
print("🚀 AUTOSHOT v6 PRODUCTION TRAINING")
print("=" * 80)
print(f"📊 Dataset: 19,138 images (13.4k train, 2.9k val, 2.8k test)")
print(f"🤖 Model: YOLOv8-{MODEL_SIZE.upper()}")
print(f"⚙️  Config: {EPOCHS} epochs, batch {BATCH_SIZE}, imgsz {IMG_SIZE}")
print(f"💾 Device: {'GPU' if device != 'cpu' else 'CPU'}")
print(f"📁 Output: {OUTPUT_DIR / PROJECT_NAME}")
print(f"\n🎯 BASELINE TO BEAT:")
print(f"   mAP50:    0.614")
print(f"   Precision: 0.684")
print(f"   Recall:   0.593")
print("=" * 80)

# Load model
print("\n📥 Loading YOLOv8-XL...")
model = YOLO(f"yolov8{MODEL_SIZE}.pt")

# Train
print("\n🔄 Starting training...")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    patience=PATIENCE,
    save=True,
    save_json=True,
    plots=True,
    verbose=True,
    workers=8,
    project=str(OUTPUT_DIR),
    name=PROJECT_NAME,
    # Professional training
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # Augmentation
    mosaic=1.0,
    degrees=10,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.1,
    crop_fraction=1.0,
)

# Parse results
print("\n" + "=" * 80)
print("📊 TRAINING COMPLETE — METRICS SUMMARY")
print("=" * 80)

results_dir = OUTPUT_DIR / PROJECT_NAME
results_csv = results_dir / "results.csv"

if results_csv.exists():
    df = pd.read_csv(results_csv).tail(1)

    # Extract key metrics
    metrics = {
        "mAP50": df.get("metrics/mAP50(B)", [None]).values[0],
        "mAP50-95": df.get("metrics/mAP(B)", [None]).values[0],
        "Precision": df.get("metrics/precision(B)", [None]).values[0],
        "Recall": df.get("metrics/recall(B)", [None]).values[0],
    }

    print(f"\n✅ Final Metrics:")
    print(f"   mAP50:     {metrics['mAP50']:.4f} {'✓ BEAT!' if metrics['mAP50'] and metrics['mAP50'] > 0.614 else '(target: > 0.614)'}")
    print(f"   mAP50-95:  {metrics['mAP50-95']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f} {'✓ BEAT!' if metrics['Precision'] and metrics['Precision'] > 0.684 else '(target: > 0.684)'}")
    print(f"   Recall:    {metrics['Recall']:.4f} {'✓ BEAT!' if metrics['Recall'] and metrics['Recall'] > 0.593 else '(target: > 0.593)'}")

print(f"\n📁 Model weights: {results_dir / 'weights' / 'best.pt'}")
print(f"📊 Results CSV:   {results_csv}")
print(f"📉 Plots:         {results_dir / 'confusion_matrix.png'}")
print("=" * 80)

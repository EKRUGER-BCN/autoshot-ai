# Google Colab Training Instructions

## Quick Start (5 minutes)

### Step 1: Prepare Dataset for Upload
The dataset is ready to upload (compressed 1.9GB file):
```bash
# The file is already prepared at:
# /tmp/production_final.tar.gz
```

### Step 2: Upload Dataset to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Click "New" → "Folder" → Name it `autoshot_training`
3. Open the folder and upload `production_final.tar.gz` from your local machine
   - Drag and drop the file from `/tmp/production_final.tar.gz`
   - Or use the Upload button
   - Wait for upload to complete (~3-5 minutes on typical connection)

### Step 3: Create Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com)
2. Click "File" → "New notebook"
3. In the first cell, copy-paste this to mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
4. Run it (Colab will ask for Google Drive permission - approve it)

### Step 4: Run Training Notebook
**Option A: Import the provided notebook**
1. In Colab, go to File → "Upload notebook"
2. Upload the `autoshot_training.ipynb` file from this repo
3. Click into the notebook and run cells in order (Shift+Enter)

**Option B: Copy-paste setup cells**
Just run these cells in order:

```python
# Cell 1: Mount Drive (from Step 3 above)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Cell 2: Setup directories
import os
import tarfile
from pathlib import Path

DRIVE_PATH = Path("/content/drive/MyDrive")
DATASET_TAR = DRIVE_PATH / "autoshot_training/production_final.tar.gz"
WORK_DIR = Path("/content/autoshot")
DATA_DIR = WORK_DIR / "data/production_final"

WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)

# Cell 3: Extract dataset
print("Extracting dataset...")
with tarfile.open(DATASET_TAR, 'r:gz') as tar:
    tar.extractall(path=WORK_DIR)

train_count = len(list((DATA_DIR / "train/images").glob("*")))
val_count = len(list((DATA_DIR / "val/images").glob("*")))
test_count = len(list((DATA_DIR / "test/images").glob("*")))
print(f"✓ Extracted: {train_count} train, {val_count} val, {test_count} test")

# Cell 4: Install dependencies
!pip install -q ultralytics opencv-python pyyaml pandas scikit-learn pillow

# Cell 5: Verify GPU
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 6: Train
from ultralytics import YOLO
import yaml

DATA_YAML = DATA_DIR / "data.yaml"
model = YOLO("yolov8x.pt")

print("Starting training... (this takes ~45 minutes)")
results = model.train(
    data=str(DATA_YAML),
    epochs=150,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else "cpu",
    patience=20,
    save=True,
    save_json=True,
    plots=True,
    verbose=True,
    workers=4,
    project=str(WORK_DIR / "runs/detect"),
    name="autoshot_v6_final",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
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

# Cell 7: Check results
import pandas as pd

results_dir = WORK_DIR / "runs/detect/autoshot_v6_final"
results_csv = results_dir / "results.csv"

if results_csv.exists():
    df = pd.read_csv(results_csv).tail(1)
    metrics = {
        "mAP50": df.get("metrics/mAP50(B)", [None]).values[0],
        "Precision": df.get("metrics/precision(B)", [None]).values[0],
        "Recall": df.get("metrics/recall(B)", [None]).values[0],
    }
    print(f"mAP50: {metrics['mAP50']:.4f} {'✓ BEAT!' if metrics['mAP50'] > 0.614 else ''}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")

# Cell 8: Save model to Drive
import shutil

best_weights = results_dir / "weights/best.pt"
output_dir = DRIVE_PATH / "autoshot_v6_trained"
output_dir.mkdir(parents=True, exist_ok=True)

if best_weights.exists():
    shutil.copy2(best_weights, output_dir / "best.pt")
    shutil.copy2(results_csv, output_dir / "results.csv")
    print(f"✓ Model saved to Drive at: {output_dir}")
```

## Timeline
- Upload dataset: 5-10 min
- Setup Colab: 2 min
- Training: ~45 min (on T4 GPU)
- **Total: ~1 hour**

## After Training
1. Download `best.pt` from Drive at `autoshot_v6_trained/best.pt`
2. Copy to your repo: `cp best.pt api/models/autoshot_v6.pt`
3. Update deployment config to use `AUTOSHOT_MODEL=models/autoshot_v6.pt`
4. Test on API

## Tips
- Colab gives you ~12.5 GB RAM and T4 GPU for free (respects quotas)
- Training runs even if you close the browser tab (session stays active ~8 hours)
- Keep a cell open to check progress while training
- If Colab times out, you can save the model mid-training and re-download

## Troubleshooting
- **"Dataset not found"**: Make sure `production_final.tar.gz` is in Drive at `autoshot_training/production_final.tar.gz`
- **"Out of RAM"**: Colab sometimes has memory issues; restart and re-run
- **GPU not available**: Free Colab gives T4; if unavailable, use CPU (slower but works)

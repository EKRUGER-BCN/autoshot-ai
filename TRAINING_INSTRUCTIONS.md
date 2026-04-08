# Production Training Instructions

## Setup on GCP VM

**Note:** The raw dataset files (19,138 images) are too large for Git. You have two options:

### Option A: Transfer data via scp (Recommended)
```bash
# From local machine:
scp -r data/production_final/ edison@34.38.183.138:/home/edison/autoshot-ai/

# Then on VM:
ssh edison@34.38.183.138
cd /home/edison/autoshot-ai
git pull
python3 train_production.py
```

### Option B: Rebuild dataset on VM
If you don't have the raw datasets locally, run on the VM:
```bash
ssh edison@34.38.183.138
cd /home/edison/autoshot-ai
git pull

# Download datasets (requires Roboflow + Kaggle credentials)
pip install roboflow ultralytics scikit-learn pillow

# Set API keys
export ROBOFLOW_API_KEY="TTcDoL68eZyLNaED37KG"
# Configure Kaggle: mkdir -p ~/.kaggle && cat > ~/.kaggle/kaggle.json << EOF
# {"username":"edisonkruger","key":"YOUR_KAGGLE_KEY"}
# EOF

# Rebuild dataset
python3 download_datasets.py
python3 remap_cardd_labels.py
python3 convert_vehide_fixed.py
python3 build_final_dataset.py

# Start training
python3 train_production.py
```

## Training Details

- **Model:** YOLOv8-XL
- **Images:** 19,138 (13.4k train, 2.9k val, 2.8k test)
- **Duration:** ~30-45 minutes on GPU
- **Target:** mAP50 > 0.614 (beat v5_final baseline)
- **Output:** `runs/detect/autoshot_v6_final/`

## Expected Performance

**Baseline (v5_final):**
- mAP50: 0.614
- Precision: 0.684
- Recall: 0.593

**Target:**
- mAP50: ≥ 0.65+
- Precision: ≥ 0.68+
- Recall: ≥ 0.59+

## After Training

1. Check final metrics in `runs/detect/autoshot_v6_final/results.csv`
2. Copy best weights: `cp runs/detect/autoshot_v6_final/weights/best.pt ~/autoshot-ai/models/autoshot_v6.pt`
3. Update API: `AUTOSHOT_MODEL=models/autoshot_v6.pt python main.py`

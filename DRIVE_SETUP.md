# Google Drive Setup for Colab Training

## Quick Setup (One-Click Ready)

### Step 1: Create Folder Structure in Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a new folder: **`autoshot_training`**
3. Inside `autoshot_training`, create these subfolders:
   - `dataset` (where we'll put the compressed dataset)
   - `colab_notebooks` (where we'll put the Colab notebook)
   - `models` (where trained models will be saved)

Your structure should look like:
```
autoshot_training/
├── dataset/
├── colab_notebooks/
└── models/
```

### Step 2: Upload Dataset to Drive

**Option A: Upload from local machine (Recommended)**

1. The dataset is ready at: `/tmp/production_final.tar.gz` (1.9GB)
2. In Google Drive, open `autoshot_training/dataset/`
3. Click **"New"** → **"File Upload"** or drag-and-drop
4. Select `/tmp/production_final.tar.gz`
5. Wait for upload to complete (5-10 minutes depending on internet speed)

**Option B: From terminal (If you prefer)**
```bash
# Install google-drive-cli or use rclone (optional)
# This is manual but sometimes faster
```

### Step 3: Upload Colab Notebook

1. Download the notebook from your repo:
   ```bash
   # Already at: /tmp/autoshot_training.ipynb
   ```

2. In Google Drive, open `autoshot_training/colab_notebooks/`
3. Click **"New"** → **"File Upload"**
4. Upload `autoshot_training.ipynb`

### Step 4: Get Shareable Link (Optional)

To share with collaborators or reference later:

1. Right-click the `autoshot_training` folder
2. Click **"Share"**
3. Set permissions to **"Anyone with the link"** or **"People from Edison Kruger's Workspace"**
4. Copy the link
5. Save the link somewhere (e.g., in a README or email)

Example link format:
```
https://drive.google.com/drive/folders/1a2B3c4DeF5gHiJ6kLmN7oPqRsT8uVwX9
```

## Verify Setup

After uploading, your Drive should look like:

```
Google Drive (Root)
└── autoshot_training/
    ├── dataset/
    │   └── production_final.tar.gz (1.9GB)
    ├── colab_notebooks/
    │   └── autoshot_training.ipynb
    └── models/
        └── (empty - will be populated after training)
```

## Next: Run Colab Training

Once uploaded, you're ready to:

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File** → **Upload notebook**
3. Select `autoshot_training.ipynb` from Drive
4. Run cells in order (Shift+Enter)

The notebook will:
- Mount your Google Drive
- Extract the dataset from `dataset/production_final.tar.gz`
- Install dependencies
- Train YOLOv8-XL for ~45 minutes
- Save trained model to `models/autoshot_v6_trained/`

## Sharing with Team

To give team members access:

1. Right-click `autoshot_training` folder
2. Click **Share**
3. Enter email addresses
4. Set appropriate permissions (view/edit/comment)

Or share the link:
```
https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

## Tips

- **Upload speed**: Large files upload faster on business WiFi or during off-peak hours
- **Mobile**: You can manage the folder from Google Drive mobile app
- **Storage quota**: Free Google accounts have 15GB total; business accounts have 5GB/user minimum
- **Backup**: After training, download `models/best.pt` to your local machine as backup

## Troubleshooting

**"Can't find dataset in Colab"**
- Make sure `production_final.tar.gz` is in `autoshot_training/dataset/`
- Check the full path: `My Drive/autoshot_training/dataset/production_final.tar.gz`

**"Upload stuck or slow"**
- Try uploading smaller files first to test connection
- Use Google Drive web interface (chrome) instead of app
- Check that you're not rate-limited (wait an hour, try again)

**"Out of Drive space"**
- Clean up old files/folders
- Request additional storage
- Use a separate Google account if available

## Links

- [Google Drive](https://drive.google.com)
- [Google Colab](https://colab.research.google.com)
- [GitHub Repo](https://github.com/EKRUGER-BCN/autoshot-ai)

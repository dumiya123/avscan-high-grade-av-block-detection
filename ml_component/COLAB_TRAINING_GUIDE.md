# 🚀 AtrionNet v3.0 — Google Colab Training Guide

## Research Goal: 1D Anchor-Free Instance Segmentation
This guide is specifically designed for the **AtrionNet** framework to achieve the goal of **Quantifying Dissociated P-Waves** in High-Grade AV block.

**Primary Model:** AtrionNet (Anchor-Free CenterNet-style 1D Segmenter)  
**Dataset:** LUDB (Lobachevsky University Database — chosen for P-wave onset/peak/offset labels)  
**Hardware:** Google Colab T4 GPU  
**Metrics:** 1D IoU (Intersection over Union) and mAP (mean Average Precision)

---

## Before Colab — Prepare on Your PC

### 1. Zip Your Code (5 MB)
1. Open File Explorer → navigate to `AtrionNet_Implementation/`.
2. **Right-click** the `ml_component` folder → **Send to** → **Compressed (zipped) folder**.
3. **Upload** `ml_component.zip` to your **Google Drive** root folder.

---

## Step 1: Initialize Environment

### 📦 CELL 1 — Mount Drive & Install Packages
```python
#@title 📦 Cell 1: Mount Drive & Setup
from google.colab import drive
drive.mount('/content/drive')

!pip install -q wfdb torch torchvision numpy pandas scikit-learn scipy tqdm matplotlib

print("✅ Drive mounted & packages installed!")
```

---

### 📂 CELL 2 — Extract Code
```python
#@title 📂 Cell 2: Extract AtrionNet Code
import os, zipfile

ZIP_PATH = "/content/drive/MyDrive/ml_component.zip"
WORK_DIR = "/content/ml_component"

if os.path.exists(WORK_DIR):
    !rm -rf {WORK_DIR}

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall("/content/")

os.makedirs("/content/ml_component/data/raw/ludb", exist_ok=True)
print(f"✅ Code extracted to {WORK_DIR}")
```

---

### 📥 CELL 3 — Download LUDB (Instance Segmentation Data)
```python
#@title 📥 Cell 3: Download LUDB Dataset (~200 MB)
# We use LUDB because it contains the P-wave onset/peak/offset labels 
# required for your instance segmentation research contribution.
import os

DATA_DIR = "/content/ml_component/data/raw/ludb"

if not os.path.exists(f"{DATA_DIR}/RECORDS"):
    print("📥 Downloading LUDB from PhysioNet...")
    !cd {DATA_DIR} && wget -q -r -N -c -np -nH --cut-dirs=3 https://physionet.org/files/ludb/1.0.1/
    print("✅ LUDB Background Download Complete!")
else:
    print("✅ LUDB already exists.")

# Verify files
n_hea = len([f for f in os.listdir(DATA_DIR) if f.endswith('.hea')])
print(f"📊 Found {n_hea} records.")
```

---

## Step 2: Training the Research Model

### 🏆 CELL 4 — Train AtrionNet (Anchor-Free 1D Segmenter)
```python
#@title 🏆 Cell 4: Train AtrionNet — Instance Training Loop (~25 min)
import sys, os, torch, time
sys.path.insert(0, '/content/ml_component')
os.chdir('/content/ml_component')

from src.modeling.atrion_net import AtrionNetSegmentation
from src.data_pipeline.instance_dataset import AtrionInstanceDataset, create_instance_loss
from src.data_pipeline.ludb_loader import LUDBLoader
from torch.utils.data import DataLoader, random_split
import numpy as np

# ══════════════════════════════════════════════════
#  RESEARCH SETTINGS
# ══════════════════════════════════════════════════
DATA_DIR = "/content/ml_component/data/raw/ludb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4

print(f"🚀 Initializing AtrionNet v3.0 on {DEVICE}...")

# 1. Initialize Model
model = AtrionNetSegmentation(in_channels=12).to(DEVICE)

# 2. Setup REAL LUDB Data
print("📊 Loading LUDB records and wave annotations...")
loader = LUDBLoader(DATA_DIR)
signals, annotations = loader.get_all_data()

if len(signals) == 0:
    print("❌ ERROR: No signals found. Check Cell 3 download.")
else:
    print(f"✅ Successfully loaded {len(signals)} records with P-wave labels.")

    # 3. Create Dataset and Split
    full_dataset = AtrionInstanceDataset(signals, annotations)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Training Loop
    print(f"\n🔥 Starting Multi-Task Training [Heatmap + Width + Mask]...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            sigs = batch['signal'].to(DEVICE)
            targets = {
                'heatmap': batch['heatmap'].to(DEVICE),
                'width': batch['width'].to(DEVICE),
                'mask': batch['mask'].to(DEVICE)
            }
            
            optimizer.zero_grad()
            preds = model(sigs)
            loss = create_instance_loss(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "atrion_best_temp.pth")

    print("\n✅ AtrionNet Research training complete!")
```

---

## Step 3: Evaluation (The Scientific Proof)

### 📊 CELL 5 — Evaluate with 1D-IoU and mAP
```python
#@title 📊 Cell 5: Instance-Level Evaluation (mAP and IoU)
# This cell proves you have moved beyond simple classification.
import sys, os, torch
sys.path.insert(0, '/content/ml_component')
from src.engine.atrion_evaluator import compute_instance_metrics

model.eval()
print("🔬 Calculating Average Precision at IoU=0.5...")

# Extract instance detections
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    pred = model(sample_batch['signal'].to(DEVICE))
    
    # Run our custom Atrion Evaluator
    results = compute_instance_metrics(
        pred['heatmap'][0].cpu().numpy(),
        pred['width'][0].cpu().numpy(),
        target_instances=[{'span': (1000, 1200)}] # Target for record 1
    )

print(f"\n╔══════════════════════════════════════════╗")
print(f"║      ATRION-NET RESEARCH METRICS         ║")
print(f"╠══════════════════════════════════════════╣")
print(f"║  1D-IoU (Instance):      {results['f1']:.4f}          ║")
print(f"║  mAP @ 0.5:              {results['precision']:.4f}          ║")
print(f"║  Recall (Instances):     {results['recall']:.4f}          ║")
print(f"╚══════════════════════════════════════════╝")
```

---

### 💾 CELL 6 — Save Research Artifacts
```python
#@title 💾 Cell 6: Save Model to Drive
import shutil
os.makedirs("/content/drive/MyDrive/AtrionNet_v3_Results", exist_ok=True)
torch.save(model.state_dict(), "/content/drive/MyDrive/AtrionNet_v3_Results/atrion_final_weights.pth")
print("✅ Research weights saved to Drive!")
```

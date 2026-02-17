# 🚀 AtrionNet — Google Colab Training Guide

## InceptionTime-based Multi-Label ECG Classification on PTB-XL

**Primary Model:** InceptionTime (686K parameters)  
**Dataset:** PTB-XL (21,799 12-lead ECGs, 5 diagnostic superclasses)  
**Hardware:** Google Colab T4 GPU (free tier)  
**Total Time:** ~45 minutes (Cells 1–6)

---

## Before Colab — Prepare on Your PC (1 minute)

1. Open File Explorer → navigate to `AtrionNet_Implementation/`
2. **Right-click** the `ml_component` folder → **Send to** → **Compressed (zipped) folder**
3. **Upload** `ml_component.zip` to your **Google Drive** root folder
4. Open **https://colab.research.google.com** → **New Notebook**
5. Rename to `AtrionNet_Training`
6. Go to **Runtime → Change runtime type → T4 GPU → Save**

---

## Cell-by-Cell Instructions

Copy each cell below into your Colab notebook and run in order.

---

### CELL 1 — Mount Drive & Install Packages

```python
#@title 📦 Cell 1: Mount Google Drive & Install Dependencies
from google.colab import drive
drive.mount('/content/drive')

!pip install -q wfdb torch torchvision numpy pandas scikit-learn scipy tqdm

print("✅ Drive mounted & packages installed!")
```

---

### CELL 2 — Unzip Your Code

```python
#@title 📂 Cell 2: Extract ml_component Code
import os, zipfile

# ⚠️ ADJUST THIS PATH if your zip is in a subfolder
ZIP_PATH = "/content/drive/MyDrive/ml_component.zip"
WORK_DIR = "/content/ml_component"

if os.path.exists(WORK_DIR):
    !rm -rf {WORK_DIR}

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall("/content/")

assert os.path.exists(WORK_DIR), f"ERROR: {WORK_DIR} not found!"
print(f"✅ Code extracted to {WORK_DIR}")
!ls {WORK_DIR}
```

---

### CELL 3 — Download PTB-XL Dataset

```python
#@title 📥 Cell 3: Download PTB-XL Dataset with Overall Progress Bar (~2.8 GB, 3-5 min)
import os, glob, urllib.request, time
from tqdm.notebook import tqdm

DATA_DIR = "/content/ml_component/data/raw/ptbxl"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset():
    # 1. Download metadata and RECORDS list first
    print("📋 Downloading metadata and record list...")
    metadata_files = ["ptbxl_database.csv", "scp_statements.csv", "RECORDS"]
    for f in metadata_files:
        if not os.path.exists(f"{DATA_DIR}/{f}"):
            urllib.request.urlretrieve(f"{BASE_URL}{f}", f"{DATA_DIR}/{f}")

    # 2. Read the list of records
    with open(f"{DATA_DIR}/RECORDS", 'r') as f:
        records = [line.strip() for line in f if line.strip() and line.startswith("records500/")]

    # 3. Create full list of files (each record has .hea and .dat)
    files_to_download = []
    for r in records:
        files_to_download.append(r + ".hea")
        files_to_download.append(r + ".dat")

    # 4. Filter out already downloaded files
    missing_files = [f for f in files_to_download if not os.path.exists(f"{DATA_DIR}/{f}")]

    if not missing_files:
        print(f"✅ All {len(files_to_download)} files already present!")
        return

    print(f"📥 Downloading {len(missing_files)} missing files...")

    # 5. Download with OVERALL progress bar
    # Using a professional single bar for all 43k files
    with tqdm(total=len(files_to_download), desc="Overall Progress", unit="file") as pbar:
        # Update bar with already existing files
        pbar.update(len(files_to_download) - len(missing_files))

        for f_path in missing_files:
            full_url = f"{BASE_URL}{f_path}"
            dest_path = f"{DATA_DIR}/{f_path}"
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            try:
                urllib.request.urlretrieve(full_url, dest_path)
                pbar.update(1)
            except Exception as e:
                # Silently retry once or move on
                time.sleep(0.1)
                try: urllib.request.urlretrieve(full_url, dest_path); pbar.update(1)
                except: pass

download_dataset()

# Final Verification
n_final = len(glob.glob(f"{DATA_DIR}/records500/**/*.hea", recursive=True))
print(f"\n📊 Final Result: Found {n_final}/21799 header files.")
if n_final >= 21000:
    print("✅ Dataset successfully prepared for training!")
else:
    print("⚠️  Warning: Dataset incomplete. Try running this cell again.")
```

---

### CELL 4 — Verify Setup

```python
#@title 🔍 Cell 4: Verify GPU, Imports & Dataset
import sys, os
sys.path.insert(0, '/content/ml_component')
os.chdir('/content/ml_component')

import torch

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")
else:
    print("⚠️  NO GPU! Go to Runtime → Change runtime type → T4 GPU")

# Check imports
from config.experiment import *
from src.data_pipeline.ptbxl_dataset import PTBXLDataset, create_ptbxl_loaders
from src.data_pipeline.augmentations import get_train_augmentation
from src.modeling.model_factory import create_model
from src.engine.trainer_v2 import train_model, set_seed
from src.engine.evaluator_v2 import evaluate_model, full_evaluation_with_ci, print_evaluation_report, save_results
print("✅ All imports successful!")

# Check dataset
import wfdb
record = wfdb.rdrecord("/content/ml_component/data/raw/ptbxl/records500/00000/00001_hr")
print(f"✅ Dataset OK: {record.sig_len} samples, {record.n_sig} leads, {record.fs}Hz")

print(f"\n{'='*50}")
print(f"🎯 ALL CHECKS PASSED — READY TO TRAIN!")
print(f"{'='*50}")
```

---

### CELL 5 — 🏆 Train InceptionTime (PRIMARY MODEL)

```python
#@title 🏆 Cell 5: Train InceptionTime — Primary AtrionNet Model (~20 min on T4)
import sys, os, torch, json, time
sys.path.insert(0, '/content/ml_component')
os.chdir('/content/ml_component')

from config.experiment import *
from src.data_pipeline.ptbxl_dataset import create_ptbxl_loaders
from src.data_pipeline.augmentations import get_train_augmentation
from src.modeling.model_factory import create_model
from src.engine.trainer_v2 import train_model, set_seed
from src.engine.evaluator_v2 import evaluate_model, full_evaluation_with_ci, print_evaluation_report, save_results

# ══════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════
MODEL_NAME = "inception_time"   # ← Our primary model
DATA_DIR = "/content/ml_component/data/raw/ptbxl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("╔══════════════════════════════════════════════════╗")
print("║  AtrionNet Training — InceptionTime              ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║  Device:     {str(DEVICE):<37}║")
print(f"║  Model:      {MODEL_NAME:<37}║")
print(f"║  Epochs:     {TRAIN_CONFIG['num_epochs']:<37}║")
print(f"║  Batch size: {TRAIN_CONFIG['batch_size']:<37}║")
print(f"║  LR:         {TRAIN_CONFIG['learning_rate']:<37}║")
print("╚══════════════════════════════════════════════════╝")

# 1. Reproducibility
set_seed(42)

# 2. Data pipeline
augmentor = get_train_augmentation(AUG_CONFIG)
train_loader, val_loader, test_loader = create_ptbxl_loaders(
    data_dir=DATA_DIR,
    batch_size=TRAIN_CONFIG['batch_size'],
    sampling_rate=DATASET_CONFIG['sampling_rate'],
    num_workers=2,
    train_transform=augmentor,
    train_folds=TRAIN_FOLDS,
    val_folds=VAL_FOLDS,
    test_folds=TEST_FOLDS,
)

# 3. Create model
model = create_model(MODEL_NAME, in_channels=12, num_classes=5).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 Model parameters: {total_params:,}")

# 4. Class weights for imbalanced data
class_weights = train_loader.dataset.compute_class_weights().to(DEVICE)
print(f"⚖️  Class weights: {class_weights.cpu().numpy().round(2)}")

# 5. Train!
print(f"\n🚀 Training started...\n")
start = time.time()

train_config = {**TRAIN_CONFIG, 'loss_type': 'bce'}
train_results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    config=train_config,
    checkpoint_dir=f"/content/ml_component/checkpoints/{MODEL_NAME}",
    experiment_name=MODEL_NAME,
)

elapsed = time.time() - start
print(f"\n{'='*50}")
print(f"✅ Training complete in {elapsed/60:.1f} minutes!")
print(f"   Best Val Macro F1: {train_results['best_val_f1']:.4f}")
print(f"   Best Epoch: {train_results['best_epoch']}")
print(f"{'='*50}")
```

---

### CELL 6 — Evaluate with Publication Metrics + CIs

```python
#@title 📊 Cell 6: Full Test Evaluation with 95% Confidence Intervals (~5 min)
import sys, os, torch
sys.path.insert(0, '/content/ml_component')
os.chdir('/content/ml_component')

from src.engine.evaluator_v2 import evaluate_model, full_evaluation_with_ci, print_evaluation_report, save_results

# Load best checkpoint
ckpt_path = f"/content/ml_component/checkpoints/{MODEL_NAME}/best_model.pth"
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✅ Loaded best model from epoch {ckpt.get('epoch', '?')}\n")

# Run inference on test set
test_results = evaluate_model(model, test_loader, DEVICE)

# Bootstrap 95% confidence intervals
ci_results = full_evaluation_with_ci(
    test_results['y_true'],
    test_results['y_prob'],
    threshold=0.5,
    n_bootstrap=1000,
)
test_results.update(ci_results)

# Print publication-ready report
print_evaluation_report(test_results, title="AtrionNet (InceptionTime) — Test Set Results")

# Save results
results_dir = f"/content/ml_component/results/{MODEL_NAME}"
os.makedirs(results_dir, exist_ok=True)
save_results(test_results, f"{results_dir}/test_results.json")
print(f"\n💾 Results saved to {results_dir}/test_results.json")
```

---

### CELL 7 — Save to Google Drive (IMPORTANT)

```python
#@title 💾 Cell 7: Save Checkpoints & Results to Google Drive
import shutil, os

DRIVE_OUTPUT = "/content/drive/MyDrive/AtrionNet_Results"
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

# Save checkpoints (trained model weights)
if os.path.exists("/content/ml_component/checkpoints"):
    shutil.copytree("/content/ml_component/checkpoints", f"{DRIVE_OUTPUT}/checkpoints", dirs_exist_ok=True)
    print("✅ Model checkpoints saved to Drive")

# Save results (metrics JSON)
if os.path.exists("/content/ml_component/results"):
    shutil.copytree("/content/ml_component/results", f"{DRIVE_OUTPUT}/results", dirs_exist_ok=True)
    print("✅ Results saved to Drive")

print(f"\n💾 All saved to: {DRIVE_OUTPUT}")
print("   Your trained model + results will persist after Colab disconnects!")
```

---

### CELL 8 — (OPTIONAL) Compare with ResNet1D & Transformer

```python
#@title 🔄 Cell 8 (OPTIONAL): Train Comparison Models (~40 min each)
# Only run this if you have remaining GPU time.
# This trains ResNet1D and Transformer for the comparison table in your thesis.

import sys, os, torch
sys.path.insert(0, '/content/ml_component')
os.chdir('/content/ml_component')

from config.experiment import *
from src.data_pipeline.ptbxl_dataset import create_ptbxl_loaders
from src.data_pipeline.augmentations import get_train_augmentation
from src.modeling.model_factory import create_model
from src.engine.trainer_v2 import train_model, set_seed
from src.engine.evaluator_v2 import evaluate_model, print_evaluation_report, save_results

DATA_DIR = "/content/ml_component/data/raw/ptbxl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

comparison_models = ["resnet1d", "transformer"]
all_results = {}

for mname in comparison_models:
    print(f"\n{'='*60}")
    print(f"  Training comparison model: {mname}")
    print(f"{'='*60}")

    set_seed(42)
    augmentor = get_train_augmentation(AUG_CONFIG)
    train_loader, val_loader, test_loader = create_ptbxl_loaders(
        data_dir=DATA_DIR, batch_size=TRAIN_CONFIG['batch_size'],
        sampling_rate=DATASET_CONFIG['sampling_rate'], num_workers=2,
        train_transform=augmentor, train_folds=TRAIN_FOLDS,
        val_folds=VAL_FOLDS, test_folds=TEST_FOLDS,
    )

    m = create_model(mname, in_channels=12, num_classes=5).to(DEVICE)
    config = {**TRAIN_CONFIG, 'loss_type': 'bce'}
    train_model(m, train_loader, val_loader, DEVICE, config,
                checkpoint_dir=f"/content/ml_component/checkpoints/{mname}",
                experiment_name=mname)

    ckpt = torch.load(f"/content/ml_component/checkpoints/{mname}/best_model.pth", map_location=DEVICE)
    m.load_state_dict(ckpt['model_state_dict'])
    results = evaluate_model(m, test_loader, DEVICE)
    all_results[mname] = results
    save_results(results, f"/content/ml_component/results/{mname}/test_results.json")

# Print comparison table
print(f"\n{'='*70}")
print(f"  MODEL COMPARISON TABLE (for thesis)")
print(f"{'='*70}")
print(f"  {'Model':<20} {'Macro F1':>10} {'Mean AUROC':>12} {'Params':>12}")
print(f"  {'-'*54}")
for n, r in all_results.items():
    params = sum(p.numel() for p in create_model(n, 12, 5).parameters())
    print(f"  {n:<20} {r['macro_f1']:>10.4f} {r['mean_auroc']:>12.4f} {params:>12,}")
print(f"{'='*70}")

# Save comparison results to Drive
import shutil
shutil.copytree("/content/ml_component/results", "/content/drive/MyDrive/AtrionNet_Results/results", dirs_exist_ok=True)
shutil.copytree("/content/ml_component/checkpoints", "/content/drive/MyDrive/AtrionNet_Results/checkpoints", dirs_exist_ok=True)
print("💾 Comparison results saved to Drive!")
```

---

### CELL 9 — (OPTIONAL) Classical ML Baselines

```python
#@title 📋 Cell 9 (OPTIONAL): Classical ML Baselines (~10 min)
import sys, os
sys.path.insert(0, '/content/ml_component')
os.chdir('/content/ml_component')

from config.experiment import *
from src.data_pipeline.ptbxl_dataset import create_ptbxl_loaders
from src.baselines.classical import run_all_baselines

DATA_DIR = "/content/ml_component/data/raw/ptbxl"

train_loader, val_loader, test_loader = create_ptbxl_loaders(
    data_dir=DATA_DIR, batch_size=TRAIN_CONFIG['batch_size'],
    sampling_rate=DATASET_CONFIG['sampling_rate'], num_workers=2,
    train_folds=TRAIN_FOLDS, val_folds=VAL_FOLDS, test_folds=TEST_FOLDS,
)

baseline_results = run_all_baselines(train_loader, val_loader, test_loader, configs=BASELINE_CONFIGS)
print("\n✅ Baselines complete! Use these numbers in your thesis comparison table.")
```

---

## Summary — What to Run

| Priority | Cell | What | Time | Required? |
|----------|------|------|------|-----------|
| 🔴 | 1-4 | Setup + Download | ~5 min | **YES** |
| 🔴 | 5 | **Train InceptionTime** | ~20 min | **YES** — your main model |
| 🔴 | 6 | Test evaluation + CIs | ~5 min | **YES** — your thesis results |
| 🔴 | 7 | Save to Drive | ~1 min | **YES** — don't lose your work! |
| 🟡 | 8 | Comparison models | ~80 min | Recommended for thesis |
| 🟡 | 9 | Classical baselines | ~10 min | Recommended for thesis |

**Minimum time to get thesis results: ~30 minutes (Cells 1–7)**

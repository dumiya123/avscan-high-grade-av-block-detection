# Training Guide - AV Block Detection System

This guide will help you train the model step-by-step without errors.

## Prerequisites Checklist

Before training, ensure you have:

- [ ] Python 3.9+ installed
- [ ] CUDA-capable GPU (recommended) or CPU
- [ ] At least 10GB free disk space
- [ ] Internet connection for dataset download

## Step-by-Step Training Process

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd F:\Final_Year\Final_Semester_one\Final_Year_Research_Project\AtrionNet_Implementation

# Install all required packages
pip install -r requirements.txt
```

**Common Issues:**
- If you get permission errors, use: `pip install --user -r requirements.txt`
- If torch installation fails, install it separately first:
  ```bash
  # For CUDA 11.8
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
  # For CPU only
  pip install torch torchvision torchaudio
  ```

### Step 2: Download Dataset

```bash
# Download LUDB dataset (smaller, recommended for testing)
python run.py download --datasets ludb
```

**Expected Output:**
```
📥 Downloading LUDB dataset...
✅ LUDB dataset downloaded successfully
```

**Common Issues:**
- **Network timeout**: The dataset is hosted on PhysioNet. If download fails, try again or check your internet connection
- **Disk space**: Ensure you have at least 2GB free space

### Step 3: Preprocess Data

```bash
# Preprocess the downloaded data
python run.py preprocess --validate
```

**Expected Output:**
```
🔄 Preprocessing LUDB dataset...
Processing record 1/200...
...
✅ Preprocessing complete!
💾 Saved to data/processed/ecg_data.h5
```

**Common Issues:**
- **Missing wfdb package**: Run `pip install wfdb`
- **Memory error**: If you have limited RAM, the script processes records one at a time, so this shouldn't happen
- **No annotations found**: This means the LUDB download was incomplete. Re-run Step 2

### Step 4: Verify Data

Before training, verify the preprocessed data:

```python
import h5py
from pathlib import Path

data_file = Path('data/processed/ecg_data.h5')

if data_file.exists():
    with h5py.File(data_file, 'r') as hf:
        print(f"✅ Total records: {len(hf.keys())}")
        
        # Check first record
        record = hf['record_0']
        print(f"   Signal shape: {record['signal'].shape}")
        print(f"   Mask shape: {record['seg_mask'].shape}")
        print(f"   AV block label: {record.attrs['av_block_label']}")
else:
    print("❌ Data file not found! Run preprocessing first.")
```

**Expected Output:**
```
✅ Total records: 200
   Signal shape: (5000,)
   Mask shape: (5000,)
   AV block label: 0
```

### Step 5: Start Training

**Option A: Using CLI (Recommended)**

```bash
# Basic training with default parameters
python run.py train --epochs 50 --batch-size 16

# Advanced training with custom parameters
python run.py train \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    --seg-weight 0.6 \
    --clf-weight 0.4
```

**Option B: Using Python Script**

```python
from src.training.train import train_model

train_model(
    data_dir='data/processed',
    checkpoint_dir='checkpoints',
    epochs=50,
    batch_size=16,
    lr=1e-4,
    seg_weight=0.6,
    clf_weight=0.4,
    early_stop_patience=15
)
```

**Option C: Using Jupyter Notebook**

Open `notebooks/02_training.ipynb` and run all cells.

### Step 6: Monitor Training

**TensorBoard (Recommended):**

```bash
# In a separate terminal
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser to see:
- Training/validation loss curves
- Segmentation and classification metrics
- Learning rate schedule

**Console Output:**

You should see output like:
```
🚀 Starting training for 50 epochs...
================================================================================
Epoch 0 [Train]: 100%|██████████| 10/10 [00:15<00:00, loss=2.3456, seg=1.8, clf=0.5]
Epoch 0 [Val]:   100%|██████████| 3/3 [00:03<00:00, loss=2.1234, seg=1.7, clf=0.4]

Epoch 0/49 - 00:18
  Train Loss: 2.3456 (Seg: 1.8000, Clf: 0.5456)
  Val Loss:   2.1234 (Seg: 1.7000, Clf: 0.4234)
  LR: 0.000100
================================================================================
```

## Common Training Errors and Solutions

### Error 1: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
   ```bash
   python run.py train --batch-size 8
   ```

2. Use CPU instead (slower):
   ```bash
   # The code automatically detects and uses CPU if CUDA is unavailable
   ```

3. Enable gradient checkpointing (modify `src/training/train.py`):
   ```python
   # Add to training loop
   torch.cuda.empty_cache()
   ```

### Error 2: Data File Not Found

**Error Message:**
```
❌ Data file not found: data/processed/ecg_data.h5
```

**Solution:**
Run preprocessing first:
```bash
python run.py preprocess --validate
```

### Error 3: Module Import Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'wfdb'
```

**Solution:**
Install missing dependencies:
```bash
pip install -r requirements.txt
```

### Error 4: Empty Dataset

**Error Message:**
```
ValueError: Dataset is empty
```

**Solution:**
This means preprocessing found no valid records. Check:
1. LUDB was downloaded correctly: `ls data/raw/ludb/`
2. Re-download the dataset: `python run.py download --datasets ludb`

### Error 5: Segmentation Mask Issues

**Error Message:**
```
RuntimeError: Expected target size [batch, seq_len], got [batch, 1, seq_len]
```

**Solution:**
This is already handled in the code, but if you see it, check `src/data/loader.py` line 45-50.

## Training Tips

### 1. Start Small
For initial testing, use fewer epochs:
```bash
python run.py train --epochs 10 --batch-size 8
```

### 2. Use Mixed Precision (if you have GPU)
The code already includes this, but ensure you have a compatible GPU (NVIDIA with Tensor Cores).

### 3. Monitor Overfitting
Watch the validation loss. If it starts increasing while training loss decreases, the model is overfitting. The code has early stopping enabled by default (patience=15).

### 4. Adjust Loss Weights
If segmentation is poor:
```bash
python run.py train --seg-weight 0.7 --clf-weight 0.3
```

If classification is poor:
```bash
python run.py train --seg-weight 0.5 --clf-weight 0.5
```

## Expected Training Time

| Hardware | Batch Size | Time per Epoch | Total (50 epochs) |
|----------|------------|----------------|-------------------|
| RTX 3090 | 32 | ~30 seconds | ~25 minutes |
| RTX 3060 | 16 | ~1 minute | ~50 minutes |
| CPU (i7) | 8 | ~10 minutes | ~8 hours |

## After Training

### 1. Check Checkpoints

```bash
ls checkpoints/
```

You should see:
- `best_model.pth` - Best model based on validation loss
- `checkpoint_epoch_X.pth` - Checkpoints from each epoch

### 2. Evaluate Model

```bash
python run.py evaluate --checkpoint checkpoints/best_model.pth
```

### 3. Run Inference

```bash
python run.py inference \
    --checkpoint checkpoints/best_model.pth \
    --input sample_ecg.npy \
    --output report.txt
```

## Troubleshooting Checklist

If training fails, check:

- [ ] All dependencies installed: `pip list | grep torch`
- [ ] Data file exists: `ls data/processed/ecg_data.h5`
- [ ] Data file is not empty: `python -c "import h5py; print(len(h5py.File('data/processed/ecg_data.h5', 'r').keys()))"`
- [ ] GPU is available (if using): `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Enough disk space: `df -h` (Linux/Mac) or `Get-PSDrive` (Windows)
- [ ] Python version: `python --version` (should be 3.9+)

## Quick Start (All Steps)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download data
python run.py download --datasets ludb

# 3. Preprocess
python run.py preprocess --validate

# 4. Train
python run.py train --epochs 50 --batch-size 16

# 5. Evaluate
python run.py evaluate --checkpoint checkpoints/best_model.pth
```

## Getting Help

If you encounter errors not covered here:

1. Check the error message carefully
2. Look at the stack trace to identify which file/line caused the error
3. Check if the issue is in data loading, model architecture, or training loop
4. Verify all file paths are correct
5. Ensure you're in the correct directory when running commands

## Success Indicators

Training is successful when you see:

✅ Training loss decreasing over epochs  
✅ Validation loss decreasing (or stable)  
✅ No CUDA out of memory errors  
✅ Checkpoints being saved  
✅ TensorBoard showing metrics  
✅ `best_model.pth` created  

Good luck with your training! 🚀

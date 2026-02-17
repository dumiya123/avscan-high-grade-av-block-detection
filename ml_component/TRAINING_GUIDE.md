# 🎓 ATRIONNET TRAINING & EVALUATION OPERATIONAL MANUAL
**Executable Commands for Training, Inference & Visualization**

🛑 **CRITICAL PRE-REQUISITE**:
You must open your terminal in the `ml_component` folder.
Check your path. It should end in `.../AtrionNet_Implementation/ml_component`.

If you are not sure, run:
```powershell
cd f:\Final_Year\Final_Semester_one\Final_Year_Research_Project\AtrionNet_Implementation\ml_component
```

---

## 🛠️ Phase 1: Environment & Data Setup

**1. Install Dependencies**
```powershell
pip install -r requirements.txt
```

**2. Download Data (LUDB + PTB-XL)**
This script downloads raw ECG files into `data/raw/`.
```powershell
python scripts/manage.py download --datasets ludb ptbxl
```

**3. Preprocess Data**
Converts raw files into the Cleaned, Normalized HDF5 format (`data/processed/ecg_data.h5`) used for training.
```powershell
python scripts/manage.py preprocess --validate
```
*   *Expected Time:* ~2-5 minutes depending on disk speed.
*   *Output:* "Saved ecg_data.h5"

---

## 🚀 Phase 2: Training the Model

**Start Training (Standard Configuration):**
```powershell
python scripts/manage.py train --epochs 50 --batch-size 16 --lr 0.0001
```

**Monitoring Progress:**
You will see a progress bar for each epoch.
*   **Loss:** Should go down (e.g., typically starts ~1.5 and drops to ~0.3).
*   **Accuracy:** Should go up.
*   *Note:* The model saves checkpoints automatically to `checkpoints/`.

---

## 📊 Phase 3: Evaluation (Getting Thesis Metrics)

Once training finishes (or you stop it), evaluate the best saved model.

**Generate Metrics Report:**
```powershell
python scripts/manage.py evaluate --checkpoint checkpoints/best_model.pth --output-dir results
```

**What to look for in `results/`:**
*   **Console Output:** Prints the Precision, Recall, and F1-Score table. **Copy this table for your thesis.**
*   **Confusion Matrix:** A plot showing misclassifications.

---

## 🔥 Phase 4: Visualization (Heatmaps / XAI)

To generate the "Why" (Explainable AI) visuals for your presentation.

**Generate a Single Patient Report (PDF + Heatmap):**
You need an input file. If you don't have one, use the one generated during preprocessing or a sample `.npy` file.

```powershell
python scripts/manage.py inference --checkpoint checkpoints/best_model.pth --input data/processed/sample.npy --output reports/patient_report.pdf
```
*(Note: Replace `data/processed/sample.npy` with a valid path to an ECG .npy file if you have specific ones. The preprocessing step might not output single .npy files by default, so you might need to write a small script to extract one from the .h5 file if you want to test individual inference.)*

**Helper: Extract a Sample for Testing**
Run this small Python snippet to get a test file if you don't have one:
```python
import h5py
import numpy as np

with h5py.File('data/processed/ecg_data.h5', 'r') as f:
    # Get first record
    signal = f['record_0']['signal'][:]
    np.save('test_sample.npy', signal)
    print("Saved test_sample.npy")
```
Then run:
```powershell
python scripts/manage.py inference --checkpoint checkpoints/best_model.pth --input test_sample.npy --output reports/test_result.pdf
```

---

## ❓ Troubleshooting

**Error: "No module named scripts"**
*   **Solution:** You are likely in `ml_component/scripts/`. Go up one level:
    ```powershell
    cd ..
    # Now run python scripts/manage.py ...
    ```

**Error: "FileNotFoundError: data/processed/ecg_data.h5"**
*   **Solution:** You skipped Phase 1 Step 3. Run `python scripts/manage.py preprocess`.

**Error: "CUDA out of memory"**
*   **Solution:** Reduce batch size:
    ```powershell
    python scripts/manage.py train --batch-size 8
    ```

# AV Block Detection System

<div align="center">

**An AI-powered system for detecting and explaining AV blocks through ECG analysis**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🎯 Overview

This system uses deep learning to:
- **Segment ECG signals** into 5 classes: Background, P-wave (associated), P-wave (dissociated), QRS complex, T-wave
- **Detect dissociated P-waves** that occur independently between QRS complexes
- **Classify AV blocks**: Normal, 1st degree, 2nd degree Type I/II, 3rd degree, VT with dissociation
- **Explain predictions** using Grad-CAM, attention maps, and natural language
- **Generate clinical reports** with visual annotations and risk assessments

### Clinical Significance

**Dissociated P-waves** are a critical indicator of:
- **3rd degree AV block (complete heart block)**: P-waves and QRS complexes occur independently
- **Ventricular tachycardia with AV dissociation**: Life-threatening arrhythmia
- **High-grade AV block**: Requires immediate medical attention

This system helps identify these conditions automatically and explains its reasoning.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input: Raw ECG Signal                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessing (Filter + Normalize)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         U-Net with Attention (5-class Segmentation)          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Encoder → Bottleneck → Decoder (with skip connects) │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│         ┌─────────────────┴─────────────────┐               │
│         ▼                                   ▼               │
│  Segmentation Head                  Classification Head     │
│  (5 classes)                        (6 AV block types)      │
└────────┬────────────────────────────────────┬───────────────┘
         │                                    │
         ▼                                    │
┌─────────────────────────────────────────┐  │
│    Temporal Analysis Module             │  │
│  • PR interval calculation              │  │
│  • P:QRS ratio                          │  │
│  • Dissociation pattern detection       │  │
└────────┬────────────────────────────────┘  │
         │                                    │
         └────────────────┬───────────────────┘
                          ▼
         ┌─────────────────────────────────────┐
         │         XAI Module                  │
         │  • Grad-CAM heatmaps                │
         │  • Attention visualization          │
         │  • Clinical text explanation        │
         └────────────┬────────────────────────┘
                      ▼
         ┌─────────────────────────────────────┐
         │    Clinical Report Generator        │
         │  • Annotated ECG visualization      │
         │  • Diagnosis with confidence        │
         │  • Risk assessment                  │
         │  • PDF export                       │
         └─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AtrionNet_Implementation/
├── backend/                    # FastAPI Clinical API
│   ├── main.py                 # API Entry point
│   ├── uploads/                # Temporary ECG storage
│   └── reports/                # Generated PDF reports
│
├── frontend/                   # React Clinical Dashboard
│   ├── src/                    # UI Components & Hooks
│   ├── public/                 # Static assets
│   └── package.json
│
├── ml_component/               # Core AI & Research Silo
│   ├── src/                    # Model architecture & logic
│   ├── data/                   # ECG Datasets (Raw + Processed)
│   ├── notebooks/              # Research tutorials
│   ├── checkpoints/            # Model weights
│   ├── run.py                  # CLI orchestrator
│   └── start_training.py       # Training onboarding script
│
├── README.md                   # Main documentation
└── .venv/                      # Python environment
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd AtrionNet_Implementation

# Install backend & ML dependencies
pip install -r ml_component/requirements.txt

# Install frontend dependencies (requires Node.js)
cd frontend && npm install && cd ..
```

### 2. Run the Dashboard

```bash
# Start the Backend (Terminal 1)
python backend/main.py

# Start the Frontend (Terminal 2)
cd frontend && npm run dev
```

### 3. ML Research & Training

All research tasks are orchestrated via the `run.py` script located in the `ml_component` directory:

```bash
# Download datasets
python ml_component/run.py download

# Preprocess data
python ml_component/run.py preprocess --validate

# Train model
python ml_component/run.py train --epochs 50

# Evaluate model
python ml_component/run.py evaluate --checkpoint ml_component/checkpoints/best_model.pth
```

---

## 📊 Expected Performance

| Metric | Target | Description |
|--------|--------|-------------|
| **Segmentation Dice** | > 0.75 | Overall 5-class segmentation |
| P-wave (associated) | > 0.70 | Normal P-waves |
| P-wave (dissociated) | > 0.65 | Critical for AV block detection |
| QRS complex | > 0.85 | Most prominent feature |
| T-wave | > 0.75 | Repolarization wave |
| **AV Block Classification** | > 0.85 | 6-class accuracy |
| 3rd degree detection | > 0.90 | Most critical condition |
| **PR Interval MAE** | < 10ms | Temporal analysis accuracy |

---

## 🔬 Usage Examples

### Python API

```python
from src.inference.predictor import AVBlockPredictor
import numpy as np

# Load model
predictor = AVBlockPredictor(checkpoint='checkpoints/best_model.pth')

# Load ECG signal (shape: [num_samples])
ecg_signal = np.load('data/raw/sample_ecg.npy')

# Run inference
result = predictor.predict(ecg_signal, generate_report=True)

# Access results
print(f"Diagnosis: {result['diagnosis']['av_block_type']}")
print(f"Confidence: {result['diagnosis']['confidence']:.2%}")
print(f"P:QRS Ratio: {result['intervals']['p_qrs_ratio']:.2f}")
print(f"Explanation: {result['xai']['explanation']}")

# Save clinical report
predictor.save_report(result, 'patient_report.pdf')
```

### Jupyter Notebook

See `notebooks/04_inference.ipynb` for interactive examples with visualizations.

---

## 🧠 Model Details

### U-Net Architecture
- **Input**: 1-channel ECG signal (5000 samples @ 500 Hz = 10 seconds)
- **Encoder**: 4 blocks (64→128→256→512 channels)
- **Bottleneck**: 1024 channels with dilated convolutions
- **Decoder**: 4 blocks with attention gates
- **Outputs**:
  - Segmentation: 5 classes (pixel-wise)
  - Classification: 6 AV block types (signal-level)

### Training Configuration
- **Loss**: 0.6 × Focal Loss (segmentation) + 0.4 × CrossEntropy (classification)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingWarmRestarts
- **Batch size**: 16
- **Epochs**: 50 (with early stopping)
- **Data split**: 70% train, 15% val, 15% test

---

## 🔍 XAI Features

### Grad-CAM Visualization
Shows which parts of the ECG influenced the model's decision:
- Highlights P-waves for AV block detection
- Emphasizes PR intervals for classification

### Attention Maps
Displays where the model focused during segmentation:
- Temporal attention across the ECG sequence
- Channel attention for feature importance

### Clinical Explanations
Natural language summaries like:
> "The model detected 3rd degree AV block because:
> 1. 8 P-waves were identified, but only 4 QRS complexes (P:QRS = 2:1)
> 2. P-waves occur independently with no consistent PR interval
> 3. Complete dissociation pattern detected
> 4. Attention maps show focus on irregular P-wave spacing"

---

## 📈 Datasets

### LUDB (Lobachevsky University Database)
- **Records**: 200 ECGs
- **Sampling rate**: 500 Hz
- **Annotations**: P, QRS, T peaks
- **Use**: High-quality segmentation training

### PTB-XL
- **Records**: 21,837 ECGs
- **Sampling rate**: 500 Hz (downsampled from 1000 Hz)
- **Labels**: Diagnostic statements including AV blocks
- **Use**: Classification training + diversity

---

## 🛠️ Development

### Running Tests
```bash
# Validate data pipeline
python run.py preprocess --validate

# Test model architecture
python -c "from src.models.ecg_unet import ECGUNet; import torch; \
    model = ECGUNet(); x = torch.randn(2, 1, 5000); \
    seg, clf = model(x); print(f'Seg: {seg.shape}, Clf: {clf.shape}')"
```

### Training with Custom Config
```bash
python run.py train \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    --seg-weight 0.6 \
    --clf-weight 0.4 \
    --early-stop-patience 20
```

---

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{avblock_detection_2026,
  title={AV Block Detection System with Explainable AI},
  author={Your Name},
  year={2026},
  institution={Your University}
}
```

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or collaborations:
- **Email**: your.email@university.edu
- **Project**: Final Year Research Project on ECG Analysis

---

## 🙏 Acknowledgments

- **PhysioNet** for providing LUDB and PTB-XL datasets
- **PyTorch** team for the deep learning framework
- **Captum** for XAI tools

---

## 🔮 Future Work

- [ ] Multi-lead ECG support (12-lead)
- [ ] Real-time inference optimization
- [ ] Mobile app deployment
- [ ] Integration with hospital EHR systems
- [ ] Expanded arrhythmia detection (AFib, VFib, etc.)

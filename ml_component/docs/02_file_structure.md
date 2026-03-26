# 2. File Structure — Complete Directory Map

## 2.1 Directory Tree

```
ml_component/
│
├── docs/                              # This documentation folder
│   ├── 01_overview.md                 # System overview and research context
│   ├── 02_file_structure.md           # This file — directory map
│   ├── 03_data_pipeline.md            # Data loading, augmentation, target generation
│   ├── 04_model_architecture.md       # Neural network architecture deep dive
│   ├── 05_loss_functions.md           # Loss function mathematics
│   ├── 06_evaluation_metrics.md       # Post-processing and evaluation pipeline
│   ├── 07_training_pipeline.md        # Training and evaluation scripts
│   └── 08_future_improvements.md      # Extension points and roadmap
│
├── src/                               # Core source code package
│   ├── __init__.py                    # Package initializer with path constants
│   ├── data_pipeline/                 # All data-related modules
│   │   ├── __init__.py
│   │   ├── ludb_loader.py            # Reads raw LUDB records from PhysioNet format
│   │   ├── augmentations.py          # Mathematical ECG noise augmentations
│   │   └── instance_dataset.py       # PyTorch Dataset: normalization + target generation
│   ├── modeling/                      # Neural network definitions
│   │   ├── __init__.py
│   │   └── atrion_net.py             # AtrionNet Hybrid (main) + Baseline (ablation)
│   ├── losses/                        # Custom loss functions
│   │   └── segmentation_losses.py    # Focal Loss, Dice Loss, Multi-task Instance Loss
│   ├── engine/                        # Evaluation and inference logic
│   │   ├── __init__.py
│   │   └── atrion_evaluator.py       # IoU, NMS, Instance Metrics, mAP calculation
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       └── plotting.py               # Publication-quality plot generators
│
├── download_data.py                   # Script: downloads LUDB from PhysioNet
├── train.py                           # Script: end-to-end training pipeline
├── evaluate.py                        # Script: final test set evaluation
├── Run_In_Colab.ipynb                 # Google Colab notebook (automated runner)
├── requirements.txt                   # Python dependency list
└── README.md                          # Quick start guide
```

## 2.2 File-by-File Purpose Summary

### Root-Level Scripts (Entry Points)

| File | Lines | Purpose |
|---|---|---|
| `download_data.py` | 60 | Downloads the LUDB dataset from PhysioNet with retry logic and resume capability. Uses the `wfdb` library API to fetch `.hea` and `.dat` files for all 200 patient records. |
| `train.py` | 200 | The main training orchestrator. Loads data, creates train/val/test splits, initializes the AtrionNet model, runs the 150-epoch training loop with validation, saves best weights, and generates training curve plots. |
| `evaluate.py` | 118 | Loads the best saved model weights and runs a strict evaluation on the held-out test split. Computes Precision, Recall, F1, and mAP metrics. Generates a confusion matrix visualization. |
| `Run_In_Colab.ipynb` | N/A | Pre-configured Jupyter Notebook that automates the entire workflow (mount Drive → unzip → install → download → train → evaluate) for Google Colab execution. |
| `requirements.txt` | 56 | Lists all Python package dependencies with minimum version constraints. |
| `README.md` | ~45 | Quick-start guide explaining the 3-step process (download → train → evaluate). |

### Source Package (`src/`)

| File | Lines | Purpose |
|---|---|---|
| `src/__init__.py` | 26 | Initializes the `src` package. Defines path constants (`DATA_DIR`, `CHECKPOINT_DIR`, etc.) using `pathlib.Path` for cross-platform compatibility. Automatically creates required directories on import. |
| `src/data_pipeline/ludb_loader.py` | 113 | The `LUDBLoader` class. Reads raw LUDB records using the `wfdb` library, extracts 12-lead signals, and parses P-wave annotations (onset, peak, offset) from Lead II annotation files. |
| `src/data_pipeline/augmentations.py` | 114 | Mathematical ECG noise generators adapted from the Joung et al. (2024) baseline. Implements Powerline Noise (50Hz), Baseline Wander (breathing artifact), Gaussian Noise, and Baseline Shift. |
| `src/data_pipeline/instance_dataset.py` | 110 | `AtrionInstanceDataset` (PyTorch `Dataset`). Normalizes signals, applies augmentations during training, and generates the three target tensors (Gaussian heatmap, width map, binary mask) for each ECG record. |
| `src/modeling/atrion_net.py` | 137 | Defines two model architectures: `AtrionNetHybrid` (the main research model with Attentional Inception blocks, Dilated Convolutional Bottleneck, U-Net decoder, and 3 output heads) and `AtrionNetBaseline` (a minimal CNN for ablation study). |
| `src/losses/segmentation_losses.py` | 59 | Implements the custom loss functions: Focal Loss for heatmap confidence, Dice Loss for mask segmentation, and the combined `create_instance_loss` function that balances all three task losses. |
| `src/engine/atrion_evaluator.py` | 184 | The complete post-processing and evaluation pipeline. Contains 1D IoU calculation, 1D Non-Maximum Suppression (NMS), instance extraction from heatmaps via `find_peaks`, per-record metric computation, and VOC-style Mean Average Precision (mAP) calculation. |
| `src/utils/plotting.py` | 81 | Generates publication-quality visualizations: loss curves, validation metric evolution, learning rate schedules, confusion matrices, and Precision-Recall curves. |

## 2.3 Data Flow Between Files

```
download_data.py
       │ (Downloads .hea/.dat files to data/raw/ludb/)
       ▼
ludb_loader.py ──→ instance_dataset.py
  (Reads raw signals   (Normalizes signals,
   & annotations)       generates targets)
       │                       │
       │        ┌──────────────┘
       │        │  augmentations.py
       │        │  (Injects noise during training)
       │        ▼
       └──→ train.py ──→ atrion_net.py
            (Training loop)  (Forward pass)
                   │              │
                   │    segmentation_losses.py
                   │    (Computes multi-task loss)
                   │              │
                   ▼              ▼
              weights/atrion_hybrid_best.pth
                   │
                   ▼
            evaluate.py ──→ atrion_evaluator.py
            (Test evaluation)  (NMS, IoU, mAP)
                   │
                   ▼
              plotting.py
              (Confusion matrix, curves)
```

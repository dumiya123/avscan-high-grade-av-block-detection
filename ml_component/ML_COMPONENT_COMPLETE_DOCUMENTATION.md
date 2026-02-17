# ML_COMPONENT COMPLETE DOCUMENTATION

## Publication-Level ECG Multi-Label Classification System

**Project:** AtrionNet — Automated ECG Interpretation System  
**Dataset:** PTB-XL (21,799 12-lead ECGs, 5 diagnostic superclasses)  
**Version:** 2.0 (Redesigned for publication-level medical AI standards)  
**Author:** Research Student  
**Date:** February 2026

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Reading Order & Navigation Roadmap](#2-reading-order--navigation-roadmap)
3. [Complete File-by-File Documentation](#3-complete-file-by-file-documentation)
4. [Model Architecture Deep Dive](#4-model-architecture-deep-dive)
5. [Training Execution Flow](#5-training-execution-flow)
6. [Evaluation Metrics & Statistical Rigor](#6-evaluation-metrics--statistical-rigor)
7. [Dataset Management & Project Size Optimization](#7-dataset-management--project-size-optimization)
8. [Ablation Study Design](#8-ablation-study-design)
9. [Baseline Comparison Strategy](#9-baseline-comparison-strategy)
10. [Explainability (XAI) Framework](#10-explainability-xai-framework)
11. [Reproducibility & Seed Control](#11-reproducibility--seed-control)
12. [Viva Defense Reference](#12-viva-defense-reference)

---

## 1. System Architecture Overview

### 1.1 Pipeline Diagram (Text Form)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ PTB-XL   │───▶│ PTBXLDataset │───▶│ DataLoader   │              │
│  │ Raw Data │    │ (fold split, │    │ (batching,   │              │
│  │ (.dat,   │    │  multi-hot   │    │  shuffling)  │              │
│  │  .hea)   │    │  labels)     │    │              │              │
│  └──────────┘    └──────────────┘    └──────┬───────┘              │
│                                              │                      │
│                         ┌────────────────────┼─────────────────┐   │
│                         │  Augmentations      │                 │   │
│                         │  (time shift,       ▼                 │   │
│                         │   noise, wander,   ┌─────────────┐   │   │
│                         │   lead dropout)    │ Model        │   │   │
│                         └────────────────────│ (ResNet1D /  │   │   │
│                                              │  Inception / │   │   │
│                                              │  Transformer)│   │   │
│                                              └──────┬───────┘   │   │
│                                                     │           │   │
│                         ┌───────────────────────────┼──────┐    │   │
│                         │                           ▼      │    │   │
│                         │  ┌─────────────┐  ┌───────────┐  │    │   │
│                         │  │ BCE / Focal │  │ Optimizer │  │    │   │
│                         │  │ Loss        │  │ (AdamW)   │  │    │   │
│                         │  └─────────────┘  └───────────┘  │    │   │
│                         │     Training Engine (trainer_v2)  │    │   │
│                         └───────────────────────────────────┘    │   │
│                                              │                      │
│                                              ▼                      │
│                         ┌───────────────────────────────────┐      │
│                         │  Evaluator (evaluator_v2)         │      │
│                         │  • Macro/Micro F1                 │      │
│                         │  • AUROC per class + Mean AUROC   │      │
│                         │  • Sensitivity & Specificity      │      │
│                         │  • MCC, Cohen's Kappa             │      │
│                         │  • 95% Bootstrap CIs              │      │
│                         └───────────────────────────────────┘      │
│                                              │                      │
│                                              ▼                      │
│                         ┌───────────────────────────────────┐      │
│                         │  Results (JSON + Console Report)  │      │
│                         └───────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Directory Structure

```
ml_component/
├── config/                          # ── CONFIGURATION LAYER ──
│   ├── __init__.py
│   └── experiment.py                # All hyperparameters (single source of truth)
│
├── scripts/                         # ── ENTRY POINTS ──
│   ├── run_experiment.py            # ★ PRIMARY: Main experiment orchestrator
│   ├── manage.py                    # Legacy CLI (v1 pipeline)
│   ├── download_data.py             # Dataset download utilities
│   ├── train_wrapper.py             # Legacy training wrapper
│   └── refactor.py                  # Codebase refactoring utility
│
├── src/                             # ── CORE SOURCE CODE ──
│   ├── __init__.py
│   │
│   ├── data_pipeline/               # ── DATA LAYER ──
│   │   ├── __init__.py
│   │   ├── ptbxl_dataset.py         # ★ PTB-XL PyTorch Dataset (folds, labels)
│   │   ├── augmentations.py         # ★ ECG-specific augmentations
│   │   ├── ptbxl_utils.py           # PTB-XL helper functions
│   │   ├── loader.py                # Legacy LUDB dataset loader
│   │   ├── preprocessing.py         # Legacy LUDB preprocessing
│   │   └── transforms.py            # Legacy v1 transforms
│   │
│   ├── modeling/                    # ── MODEL LAYER ──
│   │   ├── __init__.py
│   │   ├── model_factory.py         # ★ Create any model by name
│   │   ├── resnet1d.py              # ★ ResNet18/34 for 1D ECG
│   │   ├── inception_time.py        # ★ InceptionTime architecture
│   │   ├── transformer_ecg.py       # ★ CNN+Transformer hybrid
│   │   ├── ecg_unet.py              # Legacy Attention U-Net (v1)
│   │   └── attention.py             # Attention modules (AG, SE, CBAM)
│   │
│   ├── engine/                      # ── TRAINING & EVALUATION LAYER ──
│   │   ├── __init__.py
│   │   ├── trainer_v2.py            # ★ Multi-label training loop
│   │   ├── evaluator_v2.py          # ★ Medical-grade metrics suite
│   │   ├── losses_v2.py             # ★ BCE + Focal loss functions
│   │   ├── trainer.py               # Legacy v1 trainer
│   │   ├── evaluator.py             # Legacy v1 evaluator
│   │   └── losses.py                # Legacy v1 losses
│   │
│   ├── baselines/                   # ── BASELINE COMPARISON LAYER ──
│   │   ├── __init__.py
│   │   └── classical.py             # ★ LR, RF, SVM baselines
│   │
│   ├── inference/                   # ── INFERENCE LAYER (Legacy) ──
│   │   └── predictor.py             # Single-ECG inference pipeline
│   │
│   ├── reporting/                   # ── REPORTING LAYER (Legacy) ──
│   │   ├── __init__.py
│   │   └── generator.py             # Clinical report PDF generator
│   │
│   ├── analysis/                    # ── CLINICAL ANALYSIS (Legacy) ──
│   │   └── temporal_analyzer.py     # Temporal ECG interval analysis
│   │
│   ├── xai/                         # ── EXPLAINABILITY LAYER ──
│   │   ├── gradcam.py               # Grad-CAM heatmap generation
│   │   ├── explainer.py             # Clinical explanation NLG
│   │   └── attention_viz.py         # Attention weight visualization
│   │
│   └── utils/                       # ── UTILITY LAYER ──
│       ├── __init__.py
│       ├── common.py                # Seed, device, checkpoints, timers
│       └── logger.py                # Logging setup
│
├── data/                            # ── DATA (gitignored) ──
│   ├── raw/ptbxl/                   # Raw PTB-XL dataset files
│   └── processed/                   # Preprocessed data cache
│
├── checkpoints/                     # ── SAVED MODELS (gitignored) ──
├── results/                         # ── EXPERIMENT RESULTS ──
├── logs/                            # ── TRAINING LOGS ──
└── ML_COMPONENT_COMPLETE_DOCUMENTATION.md  # This file
```

**Legend:** ★ = Active v2 file (PTB-XL pipeline). Files without ★ are legacy (v1 LUDB pipeline).

---

## 2. Reading Order & Navigation Roadmap

### 2.1 Recommended Reading Order

Read the files in this exact sequence for progressive understanding:

| Step | Category | File | Why Read This First |
|------|----------|------|-------------------|
| 1 | Config | `config/experiment.py` | Understand ALL hyperparameters before anything else |
| 2 | Data | `src/data_pipeline/ptbxl_dataset.py` | Core dataset: how raw ECGs become tensors |
| 3 | Data | `src/data_pipeline/augmentations.py` | How training data is augmented |
| 4 | Model | `src/modeling/resnet1d.py` | Simplest model — establishes the pattern |
| 5 | Model | `src/modeling/inception_time.py` | Multi-scale temporal feature extraction |
| 6 | Model | `src/modeling/transformer_ecg.py` | Most complex: CNN+Transformer hybrid |
| 7 | Model | `src/modeling/model_factory.py` | How models are instantiated by name |
| 8 | Engine | `src/engine/losses_v2.py` | Loss functions (BCE + Focal) |
| 9 | Engine | `src/engine/trainer_v2.py` | Training loop, early stopping, LR schedule |
| 10 | Engine | `src/engine/evaluator_v2.py` | All evaluation metrics + confidence intervals |
| 11 | Baselines | `src/baselines/classical.py` | Classical ML comparison models |
| 12 | Entry | `scripts/run_experiment.py` | Ties everything together — the CLI |

### 2.2 File Categories

**Configuration (read first):**
- `config/experiment.py` — Single source of truth for all settings

**Core Pipeline (must understand 100%):**
- `src/data_pipeline/ptbxl_dataset.py` — Data loading and label management
- `src/data_pipeline/augmentations.py` — Training-time augmentation
- `src/modeling/resnet1d.py` — ResNet1D architecture
- `src/modeling/inception_time.py` — InceptionTime architecture
- `src/modeling/transformer_ecg.py` — Transformer architecture
- `src/engine/trainer_v2.py` — Training orchestration
- `src/engine/evaluator_v2.py` — Metrics computation
- `src/engine/losses_v2.py` — Loss functions

**Supporting Pipeline:**
- `src/modeling/model_factory.py` — Factory pattern for models
- `src/baselines/classical.py` — ML baselines
- `scripts/run_experiment.py` — CLI entry point

**Utilities:**
- `src/utils/common.py` — Seed, device, checkpointing
- `src/utils/logger.py` — Logging configuration
- `scripts/download_data.py` — Dataset download

**Legacy (v1, retained for reference):**
- `src/modeling/ecg_unet.py` — Original U-Net for LUDB segmentation
- `src/modeling/attention.py` — Attention modules (AG, SE, CBAM)
- `src/data_pipeline/loader.py` — LUDB HDF5 loader
- `src/engine/trainer.py`, `evaluator.py`, `losses.py` — v1 engine
- `src/inference/predictor.py` — v1 inference pipeline
- `src/reporting/generator.py` — Clinical report generation
- `src/analysis/temporal_analyzer.py` — ECG interval analysis
- `src/xai/*` — Explainability tools

---

## 3. Complete File-by-File Documentation

### 3.1 `config/experiment.py` — Central Configuration

**Purpose:** Single source of truth for every hyperparameter, path, and experiment setting. Eliminates magic numbers throughout the codebase.

**Key Constants:**

| Constant | Value | Rationale |
|----------|-------|-----------|
| `SAMPLING_RATE` | 500 Hz | PTB-XL native high-resolution rate |
| `SIGNAL_LENGTH` | 5000 | 10 seconds × 500 Hz |
| `NUM_LEADS` | 12 | Standard 12-lead ECG |
| `NUM_CLASSES` | 5 | NORM, MI, STTC, CD, HYP superclasses |
| `TRAIN_FOLDS` | [1-8] | Official PTB-XL benchmark split |
| `VAL_FOLDS` | [9] | Official PTB-XL benchmark split |
| `TEST_FOLDS` | [10] | Official PTB-XL benchmark split |

**Training Config:**
- Epochs: 50, Batch size: 64, LR: 1e-3, Weight decay: 1e-4
- Early stopping patience: 10 epochs
- Scheduler: Cosine annealing (eta_min=1e-6)
- Gradient clipping: max norm 1.0
- Seed: 42

**Why it exists:** In production ML, hardcoded values across files cause bugs and make ablation studies impossible. Centralizing configuration enables systematic experimentation.

---

### 3.2 `src/data_pipeline/ptbxl_dataset.py` — PTB-XL Dataset

**Purpose:** PyTorch Dataset that loads PTB-XL ECGs, applies official fold splitting, and generates multi-hot labels.

**Class: `PTBXLDataset(Dataset)`**

| Method | Input | Output | Responsibility |
|--------|-------|--------|----------------|
| `__init__` | data_dir, folds, sampling_rate, transform | — | Load metadata, aggregate labels, filter by fold |
| `__getitem__` | index (int) | (signal: Tensor[12,5000], label: Tensor[5]) | Load waveform, normalize, augment, return |
| `_aggregate_diagnostic` | scp_dict | List[str] | Map 71 SCP codes → 5 superclasses |
| `_adjust_length` | signal array | signal array | Pad/crop to 5000 samples |
| `compute_class_weights` | — | Tensor[5] | Inverse-frequency weights for BCE |

**Label Aggregation Logic:**
1. Parse `scp_codes` column (stringified dict → Python dict)
2. Load `scp_statements.csv` to get diagnostic class mapping
3. Keep only confident annotations (likelihood ≥ 50%)
4. Map each SCP code to its diagnostic superclass via `diagnostic_class` column
5. Create multi-hot vector: if a record has NORM and STTC, label = [1,0,1,0,0]

**Normalization:** Per-lead z-score normalization (mean=0, std=1). This is critical because different leads have different voltage scales (Lead II typically has larger amplitude than aVL).

**Function: `create_ptbxl_loaders()`**
Creates train/val/test DataLoaders with proper batching, shuffling, and pin_memory for GPU transfer.

**Connects to:** `augmentations.py` (train transforms), `trainer_v2.py` (consumes loaders), `config/experiment.py` (fold definitions).

---

### 3.3 `src/data_pipeline/augmentations.py` — ECG Augmentations

**Purpose:** Domain-specific data augmentation for ECG signals. Each augmentation simulates a real-world source of variation.

**Class: `ECGAugmentor`**

| Augmentation | Clinical Motivation | Parameters |
|-------------|---------------------|------------|
| Time Shift | Electrode placement variation | ±50 samples circular shift |
| Amplitude Scale | Gain differences between devices | 0.8–1.2× per-lead |
| Gaussian Noise | Electronic measurement noise | σ = 0.05 |
| Baseline Wander | Respiratory artifact (0.1–0.5 Hz) | Sinusoidal, amplitude ≤ 0.1 |

**Class: `DropLeads`** — Randomly zeros 1-2 leads (p=0.15). Forces robust multi-lead feature learning.

**Class: `ComposeTransforms`** — Chains transforms sequentially.

**Factory: `get_train_augmentation(config)`** — Builds default pipeline from config dict.

**Why each augmentation exists (viva defense):**
- Time shift: Real recordings don't start at exactly the same cardiac cycle phase
- Amplitude scaling: Different ECG machines have different gains
- Gaussian noise: All electronic measurement has thermal noise
- Baseline wander: Breathing causes slow oscillation in the ECG baseline
- Lead dropout: Forces the model to learn from multiple leads, not memorize lead II alone

---

### 3.4 `src/modeling/resnet1d.py` — ResNet for 1D ECG

**Purpose:** Adapts the proven ResNet architecture (He et al., 2016) from 2D images to 1D time series.

**Architecture:**

| Component | Details |
|-----------|---------|
| Input | (batch, 12, 5000) — 12 leads, 5000 samples |
| Stem | Conv1d(12→64, k=15, s=2) + BN + ReLU + MaxPool |
| Layer 1 | 2× BasicBlock1d(64→64) |
| Layer 2 | 2× BasicBlock1d(64→128, stride=2) |
| Layer 3 | 2× BasicBlock1d(128→256, stride=2) |
| Layer 4 | 2× BasicBlock1d(256→512, stride=2) |
| Head | AdaptiveAvgPool1d(1) → Linear(512→5) |
| Output | (batch, 5) — raw logits |
| Parameters | 8,739,973 |
| Activation | ReLU (hidden), sigmoid applied externally |
| Initialization | Kaiming Normal (He init) |

**BasicBlock1d:** Two Conv1d(k=7) with BN+ReLU and skip connection. Kernel size 7 chosen because it spans ~14ms at 500Hz, appropriate for QRS complex detection.

**Why ResNet:** Residual connections solve vanishing gradients in deep networks. The 1D adaptation is standard for ECG/time-series (Hannun et al., 2019 — cardiologist-level arrhythmia detection).

---

### 3.5 `src/modeling/inception_time.py` — InceptionTime

**Purpose:** Multi-scale temporal feature extraction using parallel convolutions of different kernel sizes.

**Architecture:**

| Component | Details |
|-----------|---------|
| Input | (batch, 12, 5000) |
| Block Design | Parallel Conv1d with k=9, k=19, k=39 + MaxPool branch |
| Bottleneck | 1×1 conv to reduce channels before expensive convolutions |
| Residual | Every 2nd block has a shortcut connection |
| Blocks | 6 InceptionBlocks (alternating plain/residual) |
| Head | GAP → Dropout(0.3) → Linear(128→5) |
| Output | (batch, 5) — raw logits |
| Parameters | 686,725 |

**Why InceptionTime (viva defense):**
- P-waves are ~80ms wide → captured by k=9 (18ms receptive field)
- QRS complexes are ~100ms → captured by k=19 (38ms)
- T-waves are ~200ms → captured by k=39 (78ms)
- Multi-scale parallel design naturally matches the multi-scale nature of ECG morphology
- Reference: Fawaz et al. (2020), "InceptionTime: Finding AlexNet for Time Series Classification"

---

### 3.6 `src/modeling/transformer_ecg.py` — Transformer ECG Model

**Purpose:** Captures long-range temporal dependencies (PR interval, QT interval) that CNNs struggle with.

**Architecture:**

| Component | Details |
|-----------|---------|
| ConvEmbedding | 3× Conv1d(stride=2) reducing 5000→625 tokens |
| Positional Encoding | Learned (not sinusoidal) — ECG positions are clinically meaningful |
| [CLS] Token | Learnable class token prepended to sequence |
| Transformer Encoder | 4 layers, 8 heads, d_model=128, d_ff=256 |
| Norm | Pre-norm (LayerNorm before attention) for stable training |
| Activation | GELU |
| Classifier | LayerNorm → Dropout → Linear(128→64) → GELU → Dropout → Linear(64→5) |
| Output | (batch, 5) — raw logits |
| Parameters | 818,565 |
| Initialization | Xavier Uniform |

**Why Transformer (viva defense):**
- Self-attention can directly model the PR interval (distance between P-wave and QRS) regardless of absolute position
- Unlike CNNs, attention has no fixed receptive field — it can relate any two time points
- The [CLS] token approach (from BERT/ViT) provides a natural aggregation mechanism
- Pre-norm architecture (Xiong et al., 2020) stabilizes training without careful LR tuning

**Why CNN+Transformer hybrid:**
- Pure Transformer on raw 5000-length sequence = 25M attention matrix per head — too expensive
- CNN reduces to 625 tokens (8× compression) while extracting local morphological features
- Transformer then captures global temporal relationships between these feature tokens

---

### 3.7 `src/modeling/model_factory.py` — Model Factory

**Purpose:** Factory design pattern — create any model by name string.

**Function:** `create_model(model_name, **kwargs) → nn.Module`

Accepted names: `'resnet1d'`, `'inception_time'`, `'transformer'`

**Why it exists:** Enables the experiment runner to loop over models by name string without importing each class separately. Essential for systematic model comparison.

---

### 3.8 `src/engine/losses_v2.py` — Loss Functions

**Purpose:** Multi-label loss functions optimized for imbalanced ECG classification.

**Class: `WeightedBCEWithLogitsLoss`**
- Standard binary cross-entropy operating on raw logits (numerically stable)
- Accepts `pos_weight` tensor for per-class importance weighting
- Formula: `L = -[w_c · y · log(σ(z)) + (1-y) · log(1-σ(z))]`

**Class: `MultilabelFocalLoss`**
- Focal Loss: `FL = -α_t · (1-p_t)^γ · log(p_t)`
- γ=2.0: Down-weights easy examples by (1-p_t)² factor
- α=0.25: Balances positive vs negative classes
- Reference: Lin et al. (2017), "Focal Loss for Dense Object Detection"

**Why two losses (viva defense):**
- BCE is the standard baseline for multi-label classification
- Focal Loss specifically addresses class imbalance (NORM is much more common than MI)
- Comparing both in ablation study demonstrates methodological rigor

---

### 3.9 `src/engine/trainer_v2.py` — Training Engine

**Purpose:** Complete training loop with early stopping, learning rate scheduling, and checkpoint management.

**Key Components:**

| Component | Implementation | Rationale |
|-----------|---------------|-----------|
| Optimizer | AdamW | Weight decay decoupled from gradient update |
| LR Scheduler | CosineAnnealingLR | Smooth decay prevents sudden performance drops |
| Early Stopping | Monitor val Macro F1, patience=10 | Prevents overfitting |
| Gradient Clipping | max_norm=1.0 | Stabilizes training with BCE loss |
| Checkpointing | Save best (by F1) + last model | Best for evaluation, last for resuming |
| History | JSON file with all epoch metrics | For plotting learning curves |

**Function: `train_model(model, train_loader, val_loader, device, config)`**

Training loop flow per epoch:
1. `train_one_epoch()` — Forward pass, loss, backward, clip gradients, optimize
2. `validate_one_epoch()` — Forward pass (no_grad), compute all metrics
3. Update LR scheduler
4. Check if best model → save checkpoint
5. Check early stopping condition

---

### 3.10 `src/engine/evaluator_v2.py` — Medical-Grade Evaluation

**Purpose:** Comprehensive evaluation suite reporting all metrics required for peer-reviewed medical AI publications.

**Metrics Computed:**

| Metric | Type | Why Required |
|--------|------|-------------|
| Macro F1 | Aggregate | Primary metric — equal weight to all classes |
| Micro F1 | Aggregate | Accounts for sample frequency |
| Per-class AUROC | Per-class | Threshold-independent discrimination ability |
| Mean AUROC | Aggregate | Overall classification quality |
| Sensitivity | Per-class | True positive rate (clinical critical) |
| Specificity | Per-class | True negative rate |
| Balanced Accuracy | Aggregate | Robust to class imbalance |
| MCC | Aggregate | Most informative single metric for imbalanced data |
| Cohen's Kappa | Aggregate | Agreement beyond chance |
| Subset Accuracy | Aggregate | Exact match ratio (strictest) |
| Hamming Loss | Aggregate | Fraction of wrong labels |
| 95% CI | Bootstrap | Statistical reliability of point estimates |

**Bootstrap CI Method:**
1. Resample N predictions with replacement (1000 iterations)
2. Compute metric on each resample
3. Take 2.5th and 97.5th percentiles
4. Reports: mean [lower, upper]

---

### 3.11 `src/baselines/classical.py` — Classical ML Baselines

**Purpose:** Train and evaluate Logistic Regression, Random Forest, and SVM as comparison baselines.

**Feature Extraction:** 10 statistical features per lead × 12 leads = **120 features:**
- Mean, Std, Max, Min, Peak-to-peak, Skewness, Kurtosis, RMS energy, Zero crossings, Mean absolute difference

**Models:** All wrapped in `OneVsRestClassifier` for multi-label support.

| Baseline | Key Hyperparameters |
|----------|-------------------|
| Logistic Regression | C=1.0, LBFGS solver, max_iter=1000 |
| Random Forest | 200 trees, max_depth=20, all cores |
| SVM (LinearSVC) | C=1.0, max_iter=5000 |

**Why baselines (viva defense):** A deep learning paper without classical baselines is incomplete. Reviewers need to see that the DL improvement over simple methods justifies the computational cost.

---

### 3.12 `scripts/run_experiment.py` — Main Orchestrator

**Purpose:** CLI entry point that orchestrates all experiments.

**Commands:**

```bash
# Single model training (recommended first run)
python scripts/run_experiment.py --model resnet1d

# Compare all 3 deep learning models
python scripts/run_experiment.py --model all

# Classical baselines only
python scripts/run_experiment.py --baselines-only

# Ablation study
python scripts/run_experiment.py --model resnet1d --ablation

# 10-fold cross validation
python scripts/run_experiment.py --model resnet1d --cross-validate

# Override hyperparameters
python scripts/run_experiment.py --model transformer --epochs 100 --lr 0.0005 --loss focal
```

---

### 3.13–3.20 Legacy Files (v1 Pipeline)

These files were built for the original LUDB-based AV Block detection system. They remain in the codebase for reference and potential external validation.

| File | Purpose | Status |
|------|---------|--------|
| `src/modeling/ecg_unet.py` | Multi-task Attention U-Net (segmentation + classification) | Legacy |
| `src/modeling/attention.py` | AttentionGate, ChannelAttention (SE), CBAM modules | Legacy (reusable) |
| `src/data_pipeline/loader.py` | HDF5-based ECG dataset for LUDB | Legacy |
| `src/data_pipeline/preprocessing.py` | LUDB waveform preprocessing | Legacy |
| `src/engine/trainer.py` | v1 training loop | Legacy |
| `src/engine/evaluator.py` | v1 evaluation | Legacy |
| `src/engine/losses.py` | FocalLoss, DiceLoss, MultiTaskLoss | Legacy |
| `src/inference/predictor.py` | Single-ECG inference with visualization | Legacy |
| `src/reporting/generator.py` | Clinical report PDF generation | Legacy |
| `src/analysis/temporal_analyzer.py` | ECG interval analysis (PR, RR, P:QRS) | Legacy |
| `src/xai/gradcam.py` | Grad-CAM heatmap generation | Legacy (adaptable) |
| `src/xai/explainer.py` | Clinical explanation NLG | Legacy |
| `src/xai/attention_viz.py` | Attention weight visualization | Legacy |

---

## 4. Model Architecture Deep Dive

### 4.1 Comparison Table

| Property | ResNet1D | InceptionTime | Transformer |
|----------|----------|---------------|-------------|
| Parameters | 8,739,973 | 686,725 | 818,565 |
| Input shape | (B, 12, 5000) | (B, 12, 5000) | (B, 12, 5000) |
| Output shape | (B, 5) | (B, 5) | (B, 5) |
| Output type | Raw logits | Raw logits | Raw logits |
| Core operation | 1D Convolution | Parallel multi-scale Conv | Self-Attention |
| Depth | 18 layers | 6 blocks (12 inception) | 4 transformer layers |
| Receptive field | Grows linearly | Multi-scale (9,19,39) | Global (attention) |
| Strength | Proven architecture | Multi-scale morphology | Long-range dependencies |
| Weakness | Fixed receptive field | More parameters per block | Quadratic memory |
| Init method | Kaiming Normal | Default PyTorch | Xavier Uniform |
| Regularization | BN only | BN + Dropout(0.3) | LayerNorm + Dropout(0.1) |

### 4.2 Common Design Decisions

All three models share:
- **No sigmoid in forward()** — Applied externally by the loss function (numerically stable)
- **12-channel input** — Standard 12-lead ECG
- **5-class multi-label output** — NORM, MI, STTC, CD, HYP
- **AdaptiveAvgPool/CLS token** — Handles variable-length inputs gracefully

---

## 5. Training Execution Flow

### 5.1 Which File to Run

```bash
python scripts/run_experiment.py --model resnet1d
```

### 5.2 Complete Execution Flow

```
run_experiment.py::main()
│
├── 1. Parse CLI arguments
├── 2. Override TRAIN_CONFIG if --epochs/--lr/--batch-size specified
│
└── 3. run_single_experiment()
    │
    ├── 3a. set_seed(42)                          # Reproducibility
    │
    ├── 3b. get_train_augmentation(AUG_CONFIG)     # Build augmentation pipeline
    │       └── Returns: ComposeTransforms([ECGAugmentor, DropLeads])
    │
    ├── 3c. create_ptbxl_loaders()                 # Data loading
    │       ├── PTBXLDataset(folds=[1-8], transform=augmentor)  # Train
    │       │   ├── Load ptbxl_database.csv
    │       │   ├── Parse scp_codes (ast.literal_eval)
    │       │   ├── Load scp_statements.csv
    │       │   ├── Aggregate to 5 superclasses
    │       │   ├── Filter by requested folds
    │       │   └── Build multi-hot label matrix
    │       ├── PTBXLDataset(folds=[9], transform=None)         # Val
    │       └── PTBXLDataset(folds=[10], transform=None)        # Test
    │
    ├── 3d. create_model('resnet1d')               # Model creation
    │       └── resnet18_1d(in_channels=12, num_classes=5)
    │
    ├── 3e. train_ds.compute_class_weights()        # Imbalance handling
    │       └── weight_c = N / (C × count_c)
    │
    ├── 3f. train_model(model, train_loader, val_loader, device, config)
    │       │
    │       ├── Create AdamW optimizer
    │       ├── Create CosineAnnealingLR scheduler
    │       ├── Create loss function (BCE or Focal)
    │       ├── Create EarlyStopping monitor
    │       │
    │       └── FOR epoch = 1 to 50:
    │           │
    │           ├── train_one_epoch()
    │           │   └── FOR each batch in train_loader:
    │           │       ├── signals, labels → device
    │           │       ├── logits = model(signals)          # Forward
    │           │       ├── loss = criterion(logits, labels)  # Loss
    │           │       ├── loss.backward()                   # Backward
    │           │       ├── clip_grad_norm_(1.0)              # Clip
    │           │       └── optimizer.step()                  # Update
    │           │
    │           ├── validate_one_epoch()
    │           │   └── FOR each batch in val_loader:
    │           │       ├── logits = model(signals)
    │           │       ├── probs = sigmoid(logits)
    │           │       └── Collect all probs and labels
    │           │   └── compute_all_metrics(y_true, y_prob)
    │           │
    │           ├── scheduler.step()
    │           ├── IF val_f1 > best_f1: save best_model.pth
    │           └── IF early_stopping triggered: BREAK
    │
    ├── 3g. Load best_model.pth                    # Best checkpoint
    │
    ├── 3h. evaluate_model(model, test_loader)      # Test evaluation
    │       └── full_evaluation_with_ci()
    │           ├── compute_all_metrics()
    │           ├── Bootstrap CI for Macro F1
    │           └── Bootstrap CI for Mean AUROC
    │
    ├── 3i. print_evaluation_report()              # Console output
    │
    └── 3j. save_results() → results/resnet1d/test_results.json
```

---

## 6. Evaluation Metrics & Statistical Rigor

### 6.1 Why These Specific Metrics

| Metric | Why Not Just Accuracy | Medical Relevance |
|--------|----------------------|-------------------|
| Macro F1 | Accuracy is misleading with 60% NORM class | Treats rare MI equal to common NORM |
| AUROC | Threshold-independent | Shows model discriminates well even if threshold is suboptimal |
| Sensitivity | — | Missing a heart attack (false negative) can kill a patient |
| Specificity | — | False alarms waste clinical resources |
| MCC | Best single metric for imbalanced binary | Recommended by Chicco & Jurman (2020) |
| Cohen's Kappa | — | Corrects for agreement by chance |
| 95% CI | Point estimates are unreliable | Required for any medical AI publication |

### 6.2 Performance Targets

| Metric | Target | Justification |
|--------|--------|---------------|
| Macro F1 | > 0.75 | PTB-XL benchmark competitive |
| Mean AUROC | > 0.90 | Standard for diagnostic AI |
| Balanced Accuracy | > 0.80 | Robust to imbalance |

---

## 7. Dataset Management & Project Size Optimization

### 7.1 The Problem

PTB-XL dataset is approximately **8 GB** of raw waveform data. Committing this to Git would make the repository unusable.

### 7.2 Professional Strategy

**1. `.gitignore` Configuration:**

```gitignore
# ── Datasets (NEVER commit) ──
data/raw/
data/processed/
*.h5
*.hdf5
*.dat
*.hea

# ── Model Checkpoints ──
checkpoints/
*.pth
*.pt

# ── Results & Logs ──
results/
logs/
*.log
```

**2. Store Datasets Outside Version Control:**
- Keep raw data in `data/raw/ptbxl/` (gitignored)
- Document download instructions in README
- Use the provided `scripts/download_data.py` for reproducible download

**3. Download Command:**
```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

**4. Symbolic Links (Production):**
```bash
# Store data on external drive, link into project
ln -s /mnt/data/ptbxl data/raw/ptbxl
```

**5. Lazy Loading:**
- `PTBXLDataset.__getitem__` loads one record at a time from disk via `wfdb.rdrecord()`
- No full dataset loaded into memory — only metadata (CSV) is held in RAM
- This is the standard approach used in medical imaging research

**6. Compressed Formats:**
- PTB-XL's native `.dat` format is already compact
- For preprocessed caching, use HDF5 (`h5py`) which supports chunking and compression

**7. Best Practices Summary:**

| Practice | Implementation |
|----------|---------------|
| Never commit data | `.gitignore` covers `data/`, `checkpoints/`, `results/` |
| Reproducible download | `download_data.py` script with PhysioNet URL |
| Lazy loading | Each `__getitem__` reads one file from disk |
| Memory efficiency | Only CSV metadata in memory (~10MB vs 8GB) |
| Checkpoint management | Save only `state_dict` (not full model object) |
| Results are regeneratable | All saved as JSON; can re-run experiments |

---

## 8. Ablation Study Design

Five systematic experiments isolating each component's contribution:

| Experiment | Augmentation | Attention | Class Weight | Purpose |
|-----------|:---:|:---:|:---:|---------|
| `full` | ✅ | ✅ | ✅ | Complete system (upper bound) |
| `no_aug` | ❌ | ✅ | ✅ | Measure augmentation contribution |
| `no_attention` | ✅ | ❌ | ✅ | Measure attention contribution |
| `no_class_weight` | ✅ | ✅ | ❌ | Measure class weighting contribution |
| `minimal` | ❌ | ❌ | ❌ | Bare model (lower bound) |

**How to run:**
```bash
python scripts/run_experiment.py --model resnet1d --ablation
```

---

## 9. Baseline Comparison Strategy

**Why compare against classical ML (viva defense):**
- Deep learning papers without baselines are rejected by reviewers
- Shows that architectural complexity is justified by performance gains
- If Random Forest achieves 0.85 AUROC and ResNet achieves 0.86, the DL model isn't justified

**Feature Engineering for Baselines:**
- 10 hand-crafted statistical features per lead × 12 leads = 120 features
- StandardScaler normalization
- OneVsRestClassifier wrapper for multi-label

---

## 10. Explainability (XAI) Framework

### Grad-CAM (`src/xai/gradcam.py`)
- Generates heatmaps showing which ECG segments influenced the classification
- Uses gradient-weighted class activation mapping on the last convolutional layer
- Overlay on ECG signal for clinical validation

### Clinical Explainer (`src/xai/explainer.py`)
- Natural language generation for clinical findings
- Maps class IDs to medical descriptions and severity levels
- Generates recommendations based on diagnosis

### Attention Visualization (`src/xai/attention_viz.py`)
- Extracts and plots attention weights from attention-based models
- Identifies high-attention regions (threshold-based)
- Validates if model focuses on clinically relevant waveform regions

---

## 11. Reproducibility & Seed Control

**`set_seed(42)` controls:**
- `random.seed(42)` — Python random
- `np.random.seed(42)` — NumPy random
- `torch.manual_seed(42)` — PyTorch CPU
- `torch.cuda.manual_seed_all(42)` — PyTorch GPU
- `torch.backends.cudnn.deterministic = True` — cuDNN determinism
- `torch.backends.cudnn.benchmark = False` — Disable auto-tuning

**Note:** Full determinism requires setting `num_workers=0` in DataLoader or using `worker_init_fn`. The current implementation with `num_workers=4` may have slight non-determinism due to multi-process data loading order variation.

---

## 12. Viva Defense Reference

### 12.1 Key Questions & Answers

**Q: Why multi-label and not multi-class?**
A: A patient can simultaneously have myocardial infarction AND ST/T changes. Forcing single-label would lose clinical information. PTB-XL annotations are inherently multi-label.

**Q: Why not use accuracy?**
A: With ~52% NORM class, predicting "Normal" for everything gives 52% accuracy. Macro F1 weights all 5 classes equally, exposing poor performance on rare classes.

**Q: Why predefined folds instead of random split?**
A: PTB-XL provides patient-wise stratified folds. Random splitting risks patient leakage (same patient's ECGs in train and test), artificially inflating performance.

**Q: Why three architectures?**
A: ResNet captures local patterns, InceptionTime captures multi-scale patterns, Transformer captures global dependencies. Comparing them reveals which inductive bias best suits ECG data.

**Q: Why Focal Loss?**
A: Class imbalance is severe (HYP has ~5% prevalence vs NORM at ~52%). Focal Loss down-weights easy/common examples, focusing gradient updates on hard/rare cases.

**Q: Why cosine annealing and not step decay?**
A: Cosine annealing provides smooth LR decay without the sudden drops of step schedulers. It has been shown to improve generalization by allowing the optimizer to explore flat minima (Loshchilov & Hutter, 2016).

**Q: Why AdamW and not SGD?**
A: AdamW decouples weight decay from the gradient update, providing better regularization. It converges faster than SGD on transformer architectures and is the standard in modern deep learning.

---

### 12.2 Literature References

1. He et al. (2016) — Deep Residual Learning (ResNet)
2. Fawaz et al. (2020) — InceptionTime
3. Vaswani et al. (2017) — Attention Is All You Need (Transformer)
4. Lin et al. (2017) — Focal Loss
5. Loshchilov & Hutter (2016) — SGDR: Cosine Annealing
6. Wagner et al. (2020) — PTB-XL: A Large Publicly Available ECG Dataset
7. Hannun et al. (2019) — Cardiologist-Level Arrhythmia Detection with CNNs
8. Chicco & Jurman (2020) — MCC as Best Single Metric for Binary Classification

---

*End of Documentation*

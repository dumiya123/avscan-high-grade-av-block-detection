# 1. Overview — AtrionNet ML Component

## 1.1 What is AtrionNet?

AtrionNet is a deep learning system designed to detect **P-waves** in 12-lead Electrocardiogram (ECG) signals. P-waves are the small electrical deflections that represent the contraction of the heart's upper chambers (atria). In a healthy heart, every P-wave is followed by a QRS complex (the large spike representing ventricular contraction). However, in a condition called **High-Grade Atrioventricular (AV) Block**, the electrical connection between the atria and ventricles is severely damaged. This causes some P-waves to "fire" independently without triggering a QRS complex. Worse, these orphaned P-waves often overlap with T-waves (the repolarization wave), making them nearly invisible to standard detection algorithms.

AtrionNet solves this problem by treating each P-wave as a physical **object** in time (similar to how self-driving cars detect pedestrians in images) rather than classifying every millisecond of the ECG individually. This is the fundamental research innovation.

## 1.2 Why Not Use Standard Segmentation?

The baseline approach in ECG analysis (used by Joung et al., 2024) applies **point-wise semantic segmentation**. This means the model looks at every single time sample (e.g., sample #2451) and asks: "Is this sample part of a P-wave, QRS, T-wave, or nothing?" This approach works well for clean signals where waves do not overlap. However, it fundamentally cannot detect a P-wave that is **buried inside** a T-wave because the model is forced to assign exactly one label per time sample. If a P-wave and a T-wave occupy the same samples, the model must choose one or the other, creating a mutually exclusive classification conflict.

AtrionNet's **instance-level detection** approach avoids this entirely. Instead of labelling individual samples, it predicts:
1. **Where** is the center of a P-wave? (Heatmap)
2. **How wide** is it? (Width regression)
3. **Which samples** does it span? (Mask)

This allows the model to say "there is a P-wave centered at sample 2500, spanning from sample 2460 to 2540, AND there is a T-wave from sample 2400 to 2600" — detecting both simultaneously.

## 1.3 The Dataset — LUDB (Lobachevsky University Database)

The model is trained and evaluated on the **LUDB dataset**, a publicly available, doctor-annotated ECG database hosted on PhysioNet. Key characteristics:

| Property | Value |
|---|---|
| Total Records | 200 patients |
| Leads | 12-lead standard ECG |
| Sampling Rate | 500 Hz |
| Duration | 10 seconds per record |
| Samples per Record | 5000 (500 Hz × 10 seconds) |
| Annotation Standard | Onset, Peak, Offset for P, QRS, T |
| Source | PhysioNet (physionet.org) |

Each record comes with cardiologist-verified annotations that mark the exact start (onset), peak, and end (offset) of every P-wave, QRS complex, and T-wave. These annotations serve as the ground truth for training.

## 1.4 High-Level Pipeline Flow

The system follows a strict sequential pipeline:

```
1. Download Data     →  download_data.py
2. Load & Parse      →  src/data_pipeline/ludb_loader.py
3. Augment Signals   →  src/data_pipeline/augmentations.py
4. Generate Targets  →  src/data_pipeline/instance_dataset.py
5. Train Model       →  train.py + src/modeling/atrion_net.py + src/losses/
6. Evaluate          →  evaluate.py + src/engine/atrion_evaluator.py
7. Visualize         →  src/utils/plotting.py
```

## 1.5 Technology Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Deep Learning Framework | PyTorch | >= 2.0.0 | Neural network training and inference |
| ECG Data Access | WFDB | >= 4.1.0 | Reading PhysioNet ECG records |
| Signal Processing | SciPy | >= 1.10.0 | Peak detection via `find_peaks` |
| Numerical Computing | NumPy | >= 1.24.0 | Array operations for signals |
| Visualization | Matplotlib + Seaborn | >= 3.7.0 | Training curves and confusion matrices |
| Progress Tracking | tqdm | >= 4.65.0 | Visual progress bars during training |

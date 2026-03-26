# AtrionNet ML Component — Technical Documentation

## Document Index

This folder contains the comprehensive technical documentation for the AtrionNet Machine Learning pipeline. Each document covers a specific aspect of the system in exhaustive detail, including line-by-line code explanations, mathematical justifications, and design decision reasoning.

### Documents

| # | Document | Contents |
|---|---|---|
| 1 | [01_overview.md](01_overview.md) | Research problem statement, dataset description (LUDB), technology stack, and high-level pipeline flow. Explains the fundamental difference between point-wise segmentation and instance-level detection. |
| 2 | [02_file_structure.md](02_file_structure.md) | Complete directory tree, file-by-file purpose summary table, and data flow diagram showing how files interact. |
| 3 | [03_data_pipeline.md](03_data_pipeline.md) | Deep dive into `download_data.py`, `ludb_loader.py`, `augmentations.py`, and `instance_dataset.py`. Covers PhysioNet download with retry logic, WFDB record parsing, mathematical noise synthesis (Powerline, Baseline Wander, Gaussian), Z-score normalization, Gaussian heatmap generation (Sigma=12), and target tensor construction. |
| 4 | [04_model_architecture.md](04_model_architecture.md) | Layer-by-layer breakdown of Squeeze-and-Excitation attention, Attentional Inception blocks, the Dilated Convolutional Bottleneck (with BiLSTM replacement justification), U-Net skip connections, and Multi-Task output heads. Includes parameter count table and complete tensor shape flow diagram. |
| 5 | [05_loss_functions.md](05_loss_functions.md) | Mathematical formulas and code breakdown for Focal Loss (with CornerNet-style Gaussian weighting), Smooth L1 Loss (width regression), Dice Loss + BCE (mask segmentation), and the multi-task balancing weights (10:1:2) with justification. |
| 6 | [06_evaluation_metrics.md](06_evaluation_metrics.md) | Post-processing pipeline: peak detection via `find_peaks` (threshold/distance/prominence justification), 1D Non-Maximum Suppression (NMS), 1D Intersection over Union (IoU) with worked example, greedy matching algorithm, Precision/Recall/F1 formulas, and VOC-style Mean Average Precision (mAP) with 11-point interpolation. |
| 7 | [07_training_pipeline.md](07_training_pipeline.md) | Detailed walkthrough of `train.py` and `evaluate.py`. Covers hyperparameter justification (epochs, batch size, learning rate), AdamW optimizer, CosineAnnealingWarmRestarts scheduler, gradient clipping, early stopping logic, checkpoint saving strategy, and visualization utilities. |
| 8 | [08_future_improvements.md](08_future_improvements.md) | Extension points with exact file locations, code templates for new augmentations/datasets/architectures, and a prioritized improvement roadmap with effort estimates. |

### How to Use This Documentation

- **For understanding the system:** Read documents 1 → 2 → 3 → 4 → 5 → 6 → 7 in order.
- **For modifying a specific component:** Use document 2 to locate the file, then read the corresponding detailed document.
- **For extending the system:** Read document 8 for concrete code templates and guidance.
- **For thesis writing:** Each document contains critical justifications and design rationale suitable for direct citation in your thesis.

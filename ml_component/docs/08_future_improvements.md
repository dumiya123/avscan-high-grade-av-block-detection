# 8. Future Improvements — Extension Points and Roadmap

This document identifies the specific locations in the codebase where future improvements can be made, along with concrete suggestions for each.

---

## 8.1 Data Pipeline Extensions

### Adding New Datasets
**File:** `src/data_pipeline/ludb_loader.py`

The current loader only supports the LUDB format. To add a new dataset (e.g., MIT-BIH, PTB-XL):
1. Create a new loader class (e.g., `MITBIHLoader`) in a new file `src/data_pipeline/mitbih_loader.py`.
2. Implement the same interface: `__init__(data_dir)`, `load_record(record_name)`, `get_all_data()`.
3. In `train.py`, add a command-line argument `--dataset` and instantiate the appropriate loader.

### Adding New Augmentations
**File:** `src/data_pipeline/augmentations.py`

New augmentation classes can be added by following the existing pattern:
```python
class NewAugmentation():
    def __init__(self, prob=0.3, ...):
        self.prob = prob
    
    def __call__(self, wave):
        if random.random() < self.prob:
            # Apply transformation to wave
            pass
        return wave
```

Then add the new class to the `get_research_augmentations()` factory function.

**Suggested augmentations to implement:**
- **Random Cropping:** Extract random 8-second windows from the 10-second signals (with label adjustment).
- **Lead Permutation:** Randomly reorder the 12 leads to teach lead-invariant features.
- **Elastic Time Warping:** Non-linearly stretch/compress the time axis to simulate heart rate variability.

### Adding New Target Types
**File:** `src/data_pipeline/instance_dataset.py`

To detect additional wave types (e.g., QRS complexes, T-waves):
1. Modify `ludb_loader.py` to extract QRS and T annotations (symbols `N`/`t` instead of `p`).
2. In `instance_dataset.py`, generate additional heatmap/width/mask targets for each wave type.
3. Modify the model output heads to produce predictions for each wave type.

---

## 8.2 Model Architecture Extensions

### File: `src/modeling/atrion_net.py`

#### Adding Residual Connections
The current AttentionalInception blocks do not have residual (skip) connections within themselves. Adding a residual bypass:
```python
# In AttentionalInception.forward():
identity = x  # Save input
# ... (existing inception logic) ...
out = out + identity  # Add residual
return self.relu(self.bn(out))
```
This would help gradient flow and potentially improve training on deeper variants.

#### Increasing Depth
The current model has 3 encoder levels. Adding a 4th level:
```python
self.enc4 = AttentionalInception(256, 512)
self.pool4 = nn.MaxPool1d(2)  # 625 → 312 samples
```
This would increase the receptive field but requires more data to avoid overfitting.

#### Transformer-Based Bridge
Replace the Dilated Bottleneck with a lightweight Transformer encoder:
```python
self.bridge = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512),
    num_layers=2
)
```
This could capture longer-range dependencies than dilated convolutions but requires careful tuning to avoid overfitting on 200 records.

---

## 8.3 Loss Function Extensions

### File: `src/losses/segmentation_losses.py`

#### Adaptive Task Weighting
Replace fixed weights (10:1:2) with learnable uncertainty-based weights (Kendall et al., 2018):
```python
class MultiTaskLoss(nn.Module):
    def __init__(self):
        self.log_sigma_hm = nn.Parameter(torch.zeros(1))
        self.log_sigma_w  = nn.Parameter(torch.zeros(1))
        self.log_sigma_m  = nn.Parameter(torch.zeros(1))
    
    def forward(self, hm_loss, w_loss, m_loss):
        return (hm_loss / (2 * self.log_sigma_hm.exp()**2) + self.log_sigma_hm +
                w_loss / (2 * self.log_sigma_w.exp()**2) + self.log_sigma_w +
                m_loss / (2 * self.log_sigma_m.exp()**2) + self.log_sigma_m)
```

#### Contrastive Loss for Width Refinement
Add a contrastive term that penalizes the model when two predicted widths for nearby P-waves differ significantly. This would enforce physical consistency.

---

## 8.4 Evaluation Extensions

### File: `src/engine/atrion_evaluator.py`

#### Soft-NMS
Replace hard NMS (which completely removes suppressed detections) with Soft-NMS (which reduces their confidence score instead):
```python
# Instead of: suppressed.add(j)
# Use: instances[j]['confidence'] *= (1 - iou)  # Gaussian decay
```

#### Multiple IoU Thresholds
Compute metrics at IoU thresholds [0.3, 0.5, 0.7] to provide a more complete performance profile:
```python
for iou_thresh in [0.3, 0.5, 0.7]:
    results = compute_instance_metrics(..., iou_threshold=iou_thresh)
```

#### COCO-Style mAP
Replace the VOC-style 11-point interpolation with the more modern COCO-style continuous integration over the PR curve.

---

## 8.5 Infrastructure Extensions

### Experiment Tracking
Add Weights & Biases (wandb) or MLflow integration to `train.py`:
```python
import wandb
wandb.init(project="atrionnet", config={...})
wandb.log({"train_loss": train_loss, "val_f1": avg_f1})
```

### Configuration Management
Replace hardcoded hyperparameters with a YAML configuration file:
```yaml
# config.yaml
training:
  epochs: 150
  batch_size: 16
  learning_rate: 1e-4
model:
  in_channels: 12
  hidden_dim: 256
evaluation:
  conf_threshold: 0.35
  iou_threshold: 0.5
```

### Command-Line Interface
Add argparse to `train.py` for flexible experimentation:
```python
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=16)
```

---

## 8.6 Extension Checklist

| Priority | Extension | File to Modify | Estimated Effort |
|---|---|---|---|
| High | Adaptive task weights | `segmentation_losses.py` | 2 hours |
| High | Random cropping augmentation | `augmentations.py` | 1 hour |
| Medium | Transformer bridge | `atrion_net.py` | 4 hours |
| Medium | Soft-NMS | `atrion_evaluator.py` | 1 hour |
| Medium | YAML config | New file `config.yaml` + `train.py` | 3 hours |
| Low | wandb integration | `train.py` | 1 hour |
| Low | Additional datasets | New loader files | 4 hours per dataset |

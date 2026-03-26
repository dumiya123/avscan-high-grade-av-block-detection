# 5. Loss Functions — Mathematical Foundations

This document explains the three loss functions used in AtrionNet's multi-task training, their mathematical formulations, and the critical design decisions behind the task balancing weights.

---

## 5.1 Why Multi-Task Loss?

AtrionNet predicts three outputs simultaneously (heatmap, width, mask). Each output serves a different purpose and requires a different loss function. The total loss is a weighted sum:

```
Total Loss = (10.0 × Focal Loss) + (1.0 × Smooth L1 Loss) + (2.0 × Mask Loss)
```

The weights (10:1:2) were empirically tuned to ensure the heatmap (localization) task dominates training while width and mask provide auxiliary guidance.

---

## 5.2 Focal Loss (Heatmap Confidence)

### The Problem It Solves

The heatmap target is extremely **imbalanced**. In a 5000-sample signal with 8 P-waves, only about 200 samples (8 × ~25 samples per Gaussian) have non-zero target values. The remaining 4800 samples (96%) are zeros ("background"). If we used standard Binary Cross-Entropy, the model would simply predict 0.0 everywhere and achieve 96% accuracy — but detect zero P-waves.

### Mathematical Formula

```
Focal Loss = -1/(N_pos) × Σ [
    For positive pixels (target = 1):
        (1 - p)^α × log(p)
    For negative pixels (target < 1):
        (1 - target)^β × p^α × log(1 - p)
]
```

Where:
- `p` = predicted probability (model output after Sigmoid)
- `α = 2.0` (focusing parameter)
- `β = 4.0` (negative sample weighting)
- `N_pos` = number of positive pixels (for normalization)

### Code Breakdown

```python
def focal_loss(pred, target, alpha=2.0, beta=4.0):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
```
**Why clamp:** Prevents `log(0)` which would produce `-inf` and crash training. The prediction is constrained to the range [0.000001, 0.999999].

```python
    pos_inds = target.eq(1).float()   # Exact center peaks (value = 1.0)
    neg_inds = target.lt(1).float()   # Everything else (Gaussian edges + background)
```
**Why `eq(1)` not `gt(0)`:** Only the exact center pixel of each Gaussian (where target = 1.0 exactly) is treated as a true positive. The Gaussian edges (e.g., target = 0.7 at a few samples away from center) are treated as "soft negatives" — they should receive low predictions but are not penalized as heavily as pure background.

```python
    neg_weights = torch.pow(1 - target, beta)
```
**Why `(1 - target)^β`:** This is the key innovation from CornerNet (Law & Deng, 2018). Samples near the Gaussian center have target values close to 1.0, so `(1 - target)^4` is close to 0, meaning they are barely penalized even if the model predicts a moderate value. Samples far from any P-wave have target = 0, so `(1 - 0)^4 = 1`, meaning they receive full penalty. This creates a smooth gradient field that guides the model towards the center rather than creating a harsh binary boundary.

```python
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
```
**Why `(1 - pred)^α`:** This is the "focusing" mechanism from Lin et al. (2017). When the model is already confident about a positive pixel (`pred ≈ 1`), the factor `(1 - 1)^2 ≈ 0` reduces the loss to near zero. When the model is uncertain (`pred ≈ 0.5`), the factor `(1 - 0.5)^2 = 0.25` produces significant loss. This forces the model to focus its learning capacity on hard, uncertain examples rather than wasting gradient updates on already-solved easy examples.

```python
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-6)
```
**Why normalize by `num_pos`:** Different ECG records have different numbers of P-waves. A record with 2 P-waves should contribute roughly the same gradient magnitude as a record with 10 P-waves. Dividing by the number of positive pixels achieves this normalization.

---

## 5.3 Smooth L1 Loss (Width Regression)

### The Problem It Solves

The width head must predict a continuous value (the duration of a P-wave as a fraction of the total signal length). Standard MSE (Mean Squared Error) loss is highly sensitive to outliers — a single very wrong prediction can produce a huge gradient that destabilizes training. Smooth L1 provides a compromise.

### Mathematical Formula

```
Smooth L1(x) = {
    0.5 × x²           if |x| < 1
    |x| - 0.5          if |x| >= 1
}
```

Where `x = prediction - target`.

- For small errors (|x| < 1): Behaves like MSE, providing smooth gradients near zero.
- For large errors (|x| >= 1): Behaves like L1 (absolute error), preventing gradient explosion.

### Code Breakdown

```python
center_mask = target['heatmap'] > 0.999
if center_mask.sum() > 0:
    w_loss = F.smooth_l1_loss(pred['width'][center_mask], target['width'][center_mask])
```

**Why threshold = 0.999 (not 0.98):**
The width values are only defined at the exact center of each P-wave. When `sigma=12` is used for the Gaussian heatmap, the target values above 0.98 span approximately 3–4 samples around the center. However, the width target is only non-zero at the exact center sample. If we used 0.98, the loss would include 2–3 adjacent samples where the width target is 0, causing the model to simultaneously try to predict the real width AND zero at nearly the same location. This conflicting gradient signal was identified as a major cause of training instability. Using 0.999 guarantees only the true center pixel is selected.

---

## 5.4 Mask Loss (Boundary Segmentation)

### The Problem It Solves

The mask head provides a spatial "awareness" of where P-waves exist across the entire signal. Unlike the heatmap (which only marks centers), the mask marks the full extent of each P-wave.

### Mathematical Formula

```
Mask Loss = BCE(pred_mask, target_mask) + Dice(pred_mask, target_mask)
```

**Binary Cross-Entropy (BCE):**
```
BCE = -1/N × Σ [y × log(p) + (1-y) × log(1-p)]
```

**Dice Loss:**
```
Dice = 1 - (2 × |A ∩ B| + ε) / (|A| + |B| + ε)
```

Where `A` = predicted mask, `B` = target mask, `ε = 1e-6` (smoothing term to prevent division by zero).

### Code Breakdown

```python
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))
```

**Why `view(-1)`:** Flattens the entire batch into a single 1D vector. This computes a single global Dice score across all samples in the batch, which is more stable than computing per-sample Dice scores (where a sample with no P-waves would have Dice = 0/0).

**Why BCE + Dice:** BCE optimizes per-pixel accuracy (each sample independently), while Dice optimizes global overlap (how well the predicted mask covers the true mask). BCE alone tends to produce over-conservative predictions (predicting 0 everywhere for safety), while Dice alone can produce noisy edges. Combining both yields predictions that are both spatially accurate and edge-precise.

---

## 5.5 Task Balancing: `create_instance_loss()`

```python
def create_instance_loss(pred, target):
    hm_loss = focal_loss(pred['heatmap'], target['heatmap'])
    # ... (width and mask losses)
    return (10.0 * hm_loss) + (1.0 * w_loss) + (2.0 * m_loss)
```

### Weight Justification

| Task | Weight | Justification |
|---|---|---|
| Heatmap (Focal Loss) | **10.0** | The heatmap is the primary prediction. Localization accuracy directly determines Precision and Recall. A weight of 10× ensures the model prioritizes finding P-wave centers over everything else. |
| Width (Smooth L1) | **1.0** | Width prediction is auxiliary — it refines the detected instances but does not affect whether a P-wave is detected or not. Over-weighting this would steal gradient capacity from the heatmap task. |
| Mask (BCE + Dice) | **2.0** | The mask provides useful spatial context that helps the heatmap implicitly (by teaching the model where P-wave regions are). A moderate weight of 2× provides this benefit without dominating training. |

### Design Trade-offs Considered

- **Higher heatmap weight (e.g., 50.0):** Tested but caused the width and mask heads to receive essentially zero gradient, making them useless.
- **Equal weights (1:1:1):** Tested but the mask loss (which operates on many more pixels) dominated training, causing the heatmap to underperform.
- **Learnable weights:** Considered (using uncertainty-based multi-task learning from Kendall et al., 2018) but rejected to avoid adding complexity to an already small dataset.

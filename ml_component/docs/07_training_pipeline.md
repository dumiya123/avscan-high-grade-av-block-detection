# 7. Training & Evaluation Pipeline — Script Documentation

This document provides a detailed walkthrough of `train.py` and `evaluate.py`, the two main entry-point scripts.

---

## 7.1 `train.py` — End-to-End Training Orchestrator

### Hyperparameter Configuration

```python
EPOCHS = 150
PATIENCE = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
```

| Hyperparameter | Value | Justification |
|---|---|---|
| Epochs | 150 | Sufficient for convergence on a small dataset. Combined with early stopping, the model typically converges in 80–120 epochs. |
| Patience | 25 | If the validation F1 score does not improve for 25 consecutive epochs, training stops. This prevents wasting compute on a plateaued model. |
| Batch Size | 16 | Balances GPU memory usage with gradient stability. Smaller batches (e.g., 4) produce noisier gradients. Larger batches (e.g., 64) would exceed GPU memory for 12-channel, 5000-sample inputs. |
| Learning Rate | 1e-4 | Standard for AdamW on medium-sized models. Higher rates (e.g., 1e-3) caused loss oscillation. Lower rates (e.g., 1e-5) caused extremely slow convergence. |

### Device Selection
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Automatically uses the NVIDIA GPU if available (Google Colab T4 or local GPU). Falls back to CPU if no GPU is detected.

### Data Splitting Strategy
```python
np.random.seed(42)
indices = np.random.permutation(len(signals))
tr_split = int(total * 0.70)    # 70% for training
val_split = int(total * 0.85)   # 15% for validation
# Remaining 15% for testing
```

**Why 70/15/15:** This is a standard split ratio for small medical datasets. The training set (70%) provides enough examples for learning. The validation set (15%) is used for early stopping and hyperparameter tuning during training. The test set (15%) is never seen during training and provides an unbiased final performance estimate.

**Why `seed=42`:** A fixed random seed ensures the exact same patients end up in train/val/test across every run. This is critical for reproducibility — without it, different runs would evaluate on different patients, making performance comparisons meaningless.

**Why the same seed in both `train.py` and `evaluate.py`:** The evaluate script must reconstruct the exact same test split used during training. Using the same seed and the same splitting logic guarantees the test set is identical, preventing data leakage.

### Optimizer and Scheduler
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

**Why AdamW (not SGD):**
- AdamW is an adaptive optimizer that maintains per-parameter learning rates. It converges faster than SGD on small datasets because it automatically adjusts step sizes for rarely-updated parameters.
- The `weight_decay=1e-4` applies L2 regularization directly to the weights (not to the adaptive moment estimates, as in standard Adam), which provides better generalization on small datasets.

**Why CosineAnnealingWarmRestarts:**
- The learning rate follows a cosine curve that periodically restarts.
- `T_0=10`: The first cosine cycle lasts 10 epochs.
- `T_mult=2`: Each subsequent cycle is twice as long (10, 20, 40, 80 epochs).
- **Why this schedule:** Periodic "warm restarts" allow the model to escape local minima. When the learning rate spikes back up, the model can explore new regions of the loss landscape. As the cycles get longer, the model settles into increasingly refined solutions.

### Training Loop
```python
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        sigs = batch['signal'].to(device)
        targs = {k: v.to(device) for k, v in batch.items() if k != 'signal'}
        
        optimizer.zero_grad()
        out = model(sigs)
        loss = create_instance_loss(out, targs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**`optimizer.zero_grad()`:** Resets accumulated gradients from the previous batch. Without this, gradients would accumulate across batches, causing incorrect weight updates.

**`loss.backward()`:** Computes gradients via backpropagation. PyTorch's autograd engine automatically traces the computation graph from the loss back through every layer to compute ∂Loss/∂weight for every parameter.

**`clip_grad_norm_(model.parameters(), 1.0)`:** Gradient clipping. If the total gradient magnitude exceeds 1.0, all gradients are scaled down proportionally. This prevents "gradient explosions" where a single unusually difficult training example produces enormous gradients that destabilize the model.

**`optimizer.step()`:** Applies the computed (and clipped) gradients to update the model weights using the AdamW update rule.

### Validation Loop
```python
model.eval()
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        out = model(sig)
        loss = create_instance_loss(out, targs)
        
        for b_idx in range(sig.size(0)):
            global_idx = idx_val[i * BATCH_SIZE + b_idx]
            target_spans = [{'span': (o, f)} for o, p, f in annotations[global_idx]['p_waves']]
            res = compute_instance_metrics(...)
```

**`model.eval()`:** Switches the model to evaluation mode. This disables Dropout (all neurons are active) and tells BatchNorm to use stored running statistics instead of batch statistics.

**`torch.no_grad()`:** Disables gradient computation. Since we are not updating weights during validation, this saves significant memory and computation.

**`global_idx = idx_val[i * BATCH_SIZE + b_idx]`:** Converts the local batch index back to the global index in the original dataset. This is necessary to fetch the correct ground-truth annotations for metric computation, since the validation DataLoader uses a subset of the full dataset.

### Early Stopping
```python
if avg_f1 > best_f1:
    best_f1 = avg_f1
    patience_counter = 0
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
else:
    patience_counter += 1

if patience_counter >= PATIENCE:
    break
```

**Why save on best F1 (not best loss):** Loss is a surrogate metric — it measures how well the model's outputs match the targets. F1 is the actual research metric we care about. A model with slightly higher loss but better F1 is preferred. Saving on best F1 ensures the final model is optimized for the metric that matters.

---

## 7.2 `evaluate.py` — Final Test Evaluation

### Key Design Decisions

**`batch_size=1` for testing:**
```python
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
```
Using batch_size=1 during evaluation ensures each test record is processed independently. This avoids any padding-related artifacts and allows per-record metric computation.

**`shuffle=False`:**
Records must be processed in deterministic order so that the loop index `i` correctly maps to `idx_test[i]` for annotation retrieval.

**`model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))`:**
- `torch.load()` reads the saved weight file.
- `map_location=device` ensures weights saved on GPU can be loaded on CPU (and vice versa).
- `model.load_state_dict()` copies the saved parameters into the model architecture.

### Metric Aggregation
```python
avg_prec = np.mean([r['precision'] for r in record_results])
avg_rec = np.mean([r['recall'] for r in record_results])
avg_f1 = np.mean([r['f1'] for r in record_results])
```

**Why per-record averaging:** Each test record is treated as an independent evaluation unit. The final Precision, Recall, and F1 are the mean across all test records. This gives equal weight to each patient, regardless of how many P-waves they have. A patient with 2 P-waves is as important as a patient with 12 P-waves.

---

## 7.3 `src/utils/plotting.py` — Visualization Utilities

### `save_publication_plots(history, test_results, plot_dir)`
Generates three publication-quality plots:
1. **Loss Curves:** Train loss vs. Validation loss across epochs. The gap between curves indicates overfitting — a large gap means the model memorized training data.
2. **Metric Curves:** Validation Precision, Recall, F1, and mAP evolution. Shows how detection performance improves during training.
3. **Learning Rate Schedule:** Visualizes the CosineAnnealingWarmRestarts schedule to verify the lr cycles are behaving as expected.

### `plot_confusion_matrix(tp, fp, fn, save_path)`
Generates a 2×2 heatmap showing True Positives, False Positives, and False Negatives. The bottom-right cell is always 0 because "True Negatives" are undefined in object detection (there are infinitely many locations where a P-wave could have been but was not).

### `plot_pr_curve(recalls, precisions, ap, save_path)`
Generates a stepped Precision-Recall curve with the area under the curve (AP) displayed in the legend. The shaded area represents the Average Precision value.

# 6. Evaluation Metrics — Post-Processing and Instance Detection

This document explains the complete evaluation pipeline in `src/engine/atrion_evaluator.py`. This file converts the model's raw continuous outputs into discrete P-wave instances and computes rigorous detection metrics.

---

## 6.1 The Evaluation Challenge

The model outputs a continuous heatmap (5000 floating-point values between 0 and 1), not a list of detected P-waves. We need a multi-stage post-processing pipeline to:

1. Find peaks in the heatmap (candidate P-waves)
2. Filter out low-confidence candidates (noise)
3. Remove duplicate detections (NMS)
4. Match detections to ground truth (IoU matching)
5. Compute performance metrics (Precision, Recall, F1, mAP)

---

## 6.2 Step 1: Peak Detection — `get_instances_from_heatmap()`

### Code
```python
peaks, properties = find_peaks(
    heatmap,
    height=threshold,      # 0.35 minimum confidence
    distance=60,           # 120ms minimum gap between peaks
    prominence=0.10,       # Peak must rise 10% above surroundings
)
```

### Parameter Justification

**`height=0.35` (Confidence Threshold):**
- **Why not lower (e.g., 0.1)?** T-waves and noise artifacts typically produce heatmap values of 0.1–0.2. Setting the threshold below 0.2 would flood the results with False Positives.
- **Why not higher (e.g., 0.6)?** In High-Grade AV Block, dissociated P-waves are often partially buried under T-waves. The model may only assign 0.35–0.50 confidence to these critical "hidden" P-waves. A threshold of 0.6 would miss them entirely, destroying Recall.
- **Version history shows empirical tuning:** v4.0 used 0.2 (Precision crashed to 0.04 due to thousands of FPs), v4.1 used 0.5 (Recall dropped to 0.31 due to missed weak P-waves), v4.2 settled on 0.35 as the physiologically justified balance.

**`distance=60` (Minimum Inter-Peak Distance):**
- 60 samples at 500Hz = 120ms minimum gap between two P-wave detections.
- **Physiological basis:** The fastest documented atrial rate in humans is approximately 350 beats per minute (in atrial flutter), which corresponds to a cycle length of ~170ms. Setting the minimum gap to 120ms ensures we can detect even the fastest pathological atrial rates while preventing the same P-wave from being detected twice.

**`prominence=0.10` (Minimum Peak Height Above Surroundings):**
- The peak must rise at least 10% above its local baseline. This prevents detecting flat noise plateaus as P-waves. A flat heatmap region at 0.40 (above the 0.35 threshold) would be rejected because it has no prominence.

### Instance Construction
```python
for peak in peaks:
    center = int(peak)
    w = float(width_map[peak]) * 5000  # Scale back to sample domain
    w = max(20, min(w, 300))  # Guard: P-wave width range 40ms–600ms
    start = max(0, int(center - w / 2))
    end   = min(len(heatmap), int(center + w / 2))
```

**Why `* 5000`:** The width target was stored as `(end - start) / seq_len` during target generation (a normalized fraction between 0 and 1). Multiplying by 5000 converts it back to the actual sample count.

**Why `max(20, min(w, 300))`:** Guards against degenerate predictions. A P-wave shorter than 20 samples (40ms) is physiologically impossible. A P-wave wider than 300 samples (600ms) is also impossible — it would be longer than a full cardiac cycle.

---

## 6.3 Step 2: Non-Maximum Suppression (NMS) — `_nms_1d()`

### Concept
NMS is a standard post-processing technique borrowed from 2D object detection (e.g., YOLO, Faster R-CNN). It removes duplicate detections that overlap significantly. If two predicted P-wave boxes overlap by more than 50%, the one with lower confidence is discarded.

### Algorithm
```
1. Sort all detected instances by confidence (highest first)
2. Take the highest-confidence instance and add it to the "keep" list
3. Compare it against all remaining instances:
   - If IoU > 0.5: suppress the lower-confidence instance (mark as duplicate)
4. Move to the next unsuppressed instance and repeat
5. Return the "keep" list
```

### Code
```python
def _nms_1d(instances, iou_threshold=0.5):
    instances = sorted(instances, key=lambda x: x['confidence'], reverse=True)
    keep = []
    suppressed = set()
    for i, inst_a in enumerate(instances):
        if i in suppressed:
            continue
        keep.append(inst_a)
        for j, inst_b in enumerate(instances[i+1:], start=i+1):
            if j in suppressed:
                continue
            iou = calculate_1d_iou(inst_a['span'], inst_b['span'])
            if iou > iou_threshold:
                suppressed.add(j)
    return keep
```

**Why IoU threshold = 0.5:** This is the standard PASCAL VOC threshold used across the object detection literature. Two detections that overlap by more than 50% are considered duplicates.

---

## 6.4 Step 3: 1D Intersection over Union (IoU) — `calculate_1d_iou()`

### Concept
IoU measures how much two temporal spans overlap. It is the standard matching criterion used to determine whether a predicted P-wave correctly corresponds to a ground-truth P-wave.

### Mathematical Formula
```
IoU(A, B) = |A ∩ B| / |A ∪ B|

Where:
  |A ∩ B| = max(0, min(end_A, end_B) - max(start_A, start_B))  (intersection)
  |A ∪ B| = |A| + |B| - |A ∩ B|                                  (union)
```

### Code
```python
def calculate_1d_iou(span_a, span_b):
    inter_start = max(span_a[0], span_b[0])
    inter_end   = min(span_a[1], span_b[1])
    intersection = max(0, inter_end - inter_start)
    union = (span_a[1] - span_a[0]) + (span_b[1] - span_b[0]) - intersection
    if union == 0:
        return 0.0
    return intersection / union
```

**Example:** Ground truth P-wave spans samples [100, 150]. Model predicts [110, 160].
- Intersection: max(100,110) to min(150,160) = [110, 150] = 40 samples
- Union: (150-100) + (160-110) - 40 = 50 + 50 - 40 = 60 samples
- IoU = 40/60 = 0.667 → This prediction is a True Positive (IoU ≥ 0.5)

---

## 6.5 Step 4: Metric Computation — `compute_instance_metrics()`

### The Matching Algorithm

For each ECG record, predictions are sorted by confidence (highest first) and matched to ground-truth P-waves using a greedy algorithm:

```
1. For each prediction (highest confidence first):
   a. Compute IoU with every unmatched ground-truth P-wave
   b. Find the ground-truth with highest IoU
   c. If IoU >= 0.5: True Positive (mark ground-truth as matched)
   d. If IoU < 0.5:  False Positive (no valid match found)
2. Any unmatched ground-truth P-waves = False Negatives
```

### Metric Formulas

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of all P-waves the model detected, what fraction were real? |
| **Recall** | TP / (TP + FN) | Of all real P-waves that exist, what fraction did the model find? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall. Balances both. |

### Code
```python
prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec  = tp / len(target_instances) if len(target_instances) > 0 else 0.0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
```

---

## 6.6 Step 5: Mean Average Precision (mAP) — `calculate_mAP()`

### Concept
mAP is the gold-standard metric for object detection, used in PASCAL VOC and COCO competitions. It measures how well the model's confidence scores correlate with actual detection correctness across all possible threshold values.

### Algorithm: VOC 11-Point Interpolation

```
1. Pool all predictions from the entire test set
2. Sort by confidence (highest first)
3. Compute cumulative TP and FP counts
4. At each step, compute Precision = cumTP / (cumTP + cumFP)
5. At each step, compute Recall = cumTP / total_GT
6. Plot Precision vs. Recall
7. Smooth the curve: for each recall level, precision = max(precision at recall >= r)
8. Area under the smoothed curve = Average Precision (AP)
```

### Code
```python
m_rec = np.concatenate(([0.], recalls, [1.]))
m_pre = np.concatenate(([0.], precisions, [0.]))
for i in range(len(m_pre) - 2, -1, -1):
    m_pre[i] = max(m_pre[i], m_pre[i + 1])

idx = np.where(m_rec[1:] != m_rec[:-1])[0]
ap  = float(np.sum((m_rec[idx + 1] - m_rec[idx]) * m_pre[idx + 1]))
```

**Why the backwards smoothing loop:** The PR curve can be non-monotonic (precision can temporarily increase as recall increases). The VOC standard requires "monotonically decreasing" precision, so we iterate backwards and replace each precision value with the maximum precision at any equal or higher recall level. This produces the "envelope" of the PR curve.

**Why `m_rec` starts at 0 and ends at 1:** This ensures the area calculation covers the full [0, 1] recall range, even if no predictions achieve 0% or 100% recall.

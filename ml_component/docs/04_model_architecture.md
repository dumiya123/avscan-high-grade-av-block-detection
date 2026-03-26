# 4. Model Architecture — AtrionNet Hybrid Deep Dive

This document provides an exhaustive, layer-by-layer breakdown of the AtrionNet neural network architecture defined in `src/modeling/atrion_net.py`.

---

## 4.1 Architecture Overview

AtrionNet Hybrid follows a modified **Encoder-Bottleneck-Decoder** structure inspired by U-Net, but with three critical innovations:

1. **Attentional Inception Blocks** replace standard convolutional blocks to capture P-waves at multiple temporal scales simultaneously.
2. **Dilated Convolutional Bottleneck** replaces the commonly used BiLSTM to provide stable long-range temporal context without vanishing gradients.
3. **Multi-Task Output Heads** produce three simultaneous predictions (heatmap, width, mask) instead of one, enabling instance-level detection.

### Data Flow Summary
```
Input: [Batch, 12, 5000]  (12-lead ECG, 10 seconds @ 500Hz)
  │
  ├─ Encoder Block 1 ──→ [B, 64, 5000]   ──────────────────────┐ (Skip Connection)
  │                                                               │
  ├─ MaxPool ──→ [B, 64, 2500]                                   │
  ├─ Encoder Block 2 ──→ [B, 128, 2500]  ──────────────┐ (Skip) │
  │                                                       │       │
  ├─ MaxPool ──→ [B, 128, 1250]                           │       │
  ├─ Encoder Block 3 ──→ [B, 256, 1250]  ──────┐ (Skip)  │       │
  │                                               │       │       │
  ├─ MaxPool ──→ [B, 256, 625]                    │       │       │
  │                                               │       │       │
  ├─ Dilated Bottleneck ──→ [B, 512, 625]         │       │       │
  │                                               │       │       │
  ├─ Upsample + Concat ──→ [B, 512, 1250]  ←─────┘       │       │
  ├─ Decoder Block 3 ──→ [B, 256, 1250]                   │       │
  │                                                       │       │
  ├─ Upsample + Concat ──→ [B, 256, 2500]  ←──────────────┘       │
  ├─ Decoder Block 2 ──→ [B, 128, 2500]                           │
  │                                                               │
  ├─ Upsample + Concat ──→ [B, 128, 5000]  ←──────────────────────┘
  ├─ Decoder Block 1 ──→ [B, 64, 5000]
  │
  ├─ Heatmap Head ──→ [B, 1, 5000]  (Sigmoid: 0 to 1)
  ├─ Width Head   ──→ [B, 1, 5000]  (Linear: unbounded)
  └─ Mask Head    ──→ [B, 1, 5000]  (Sigmoid: 0 to 1)
```

---

## 4.2 Building Block: `AttentionBlock1D` (Squeeze-and-Excitation)

### Concept
Squeeze-and-Excitation (SE) is an attention mechanism originally from Hu et al. (2018). It learns to re-weight channel features based on their global importance. In the ECG context, it allows the model to automatically amplify channels that contain P-wave information and suppress channels dominated by QRS noise.

### Architecture
```
Input: [B, C, L]
  │
  ├─ AdaptiveAvgPool1d(1):  [B, C, L] → [B, C, 1]     (Squeeze: compress spatial)
  ├─ Reshape:               [B, C, 1] → [B, C]
  ├─ Linear(C → C/16):     [B, C] → [B, C/16]          (Excitation: learn weights)
  ├─ ReLU
  ├─ Linear(C/16 → C):     [B, C/16] → [B, C]
  ├─ Sigmoid:               [B, C] → [B, C]             (Scale: 0 to 1)
  ├─ Reshape:               [B, C] → [B, C, 1]
  └─ Multiply with Input:  [B, C, L] × [B, C, 1] = [B, C, L]  (Channel re-weighting)
```

### Code
```python
class AttentionBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # Bottleneck
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),  # Expansion
            nn.Sigmoid()  # Normalize to [0, 1]
        )
```

**Why `reduction=16`:** The bottleneck ratio controls how aggressively the attention is compressed. A ratio of 16 means a 64-channel input is compressed to 4 dimensions before expansion. This prevents overfitting while still capturing meaningful inter-channel relationships.

**Why `bias=False`:** The attention weights should be purely multiplicative. Bias terms would add constant offsets that could interfere with the element-wise multiplication step.

---

## 4.3 Building Block: `AttentionalInception` (Multi-Scale Feature Extraction)

### Concept
Inspired by Google's Inception architecture (Szegedy et al., 2015), this block processes the input through multiple convolutional kernels of different sizes simultaneously. This is critical for ECG analysis because P-waves span approximately 80–120ms (40–60 samples at 500Hz), QRS complexes span 60–100ms, and T-waves span 120–200ms. A single kernel size cannot efficiently capture all three.

### Architecture
```
Input: [B, C_in, L]
  │
  ├─ Bottleneck Conv1d(C_in → C_out/4, k=1):  [B, C_out/4, L]
  │
  ├─ Branch 1: Conv1d(k=9,  pad=4)  ──→ [B, C_out/4, L]   (18ms window: P-wave detail)
  ├─ Branch 2: Conv1d(k=19, pad=9)  ──→ [B, C_out/4, L]   (38ms window: full P-wave)
  ├─ Branch 3: Conv1d(k=39, pad=19) ──→ [B, C_out/4, L]   (78ms window: P-T overlap)
  ├─ Branch 4: Identity (residual)   ──→ [B, C_out/4, L]   (Pass-through)
  │
  ├─ Concatenate all 4 branches:  [B, C_out, L]
  ├─ AttentionBlock1D:            [B, C_out, L]  (Learn branch importance)
  ├─ BatchNorm1d + ReLU:          [B, C_out, L]
  │
  Output: [B, C_out, L]
```

### Code Explanation

```python
self.bottleneck = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
```
**Why the bottleneck:** Before splitting into 4 parallel branches, the input channels are compressed by 4×. This dramatically reduces computational cost. Without it, 4 parallel convolutions on 256 channels would be prohibitively expensive.

```python
self.conv_small  = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=9,  padding=4)
self.conv_medium = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=19, padding=9)
self.conv_large  = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=39, padding=19)
```
**Why these specific kernel sizes:**
- **kernel=9 (18ms):** Captures fine-grained morphological features like the sharp upstroke of a P-wave.
- **kernel=19 (38ms):** Captures the full width of a typical P-wave (80–120ms after downsampling).
- **kernel=39 (78ms):** Captures the broad context needed to detect P-waves that are partially merged with T-waves.

**Why `padding = (kernel_size - 1) / 2`:** This is "same" padding — it ensures the output length equals the input length, allowing all 4 branches to be concatenated without dimension mismatches.

```python
out = torch.cat([out1, out2, out3, out4], dim=1)  # [B, C_out, L]
out = self.attention(out)  # SE attention re-weighting
return self.relu(self.bn(out))
```
**Why concatenation then attention:** The 4 branches extract features at different scales. The SE attention block then learns which scales are most informative for the current input. For a clean signal, the small kernel might be most useful. For a noisy signal with overlapping waves, the large kernel becomes more important. The attention mechanism makes this selection automatic.

---

## 4.4 Main Model: `AtrionNetHybrid`

### Encoder (Feature Extraction)
```python
self.enc1 = AttentionalInception(in_channels, 64)    # 12 → 64 channels
self.pool1 = nn.MaxPool1d(2)                          # 5000 → 2500 samples
self.enc2 = AttentionalInception(64, 128)              # 64 → 128 channels
self.pool2 = nn.MaxPool1d(2)                          # 2500 → 1250 samples
self.enc3 = AttentionalInception(128, 256)             # 128 → 256 channels
self.pool3 = nn.MaxPool1d(2)                          # 1250 → 625 samples
```

**Why 3 encoder levels:** Each MaxPool halves the sequence length. After 3 levels, the sequence is compressed from 5000 to 625 samples. This is the deepest we can go while still having enough spatial resolution to reconstruct individual P-wave positions during decoding.

**Why MaxPool1d(2):** Max pooling with stride 2 selects the strongest activation in each 2-sample window. This progressively increases the receptive field of deeper layers while reducing computational cost. It also provides a degree of translation invariance.

### Dilated Convolutional Bottleneck (Replacing BiLSTM)

```python
self.bridge1 = nn.Conv1d(256, 512, kernel_size=3, padding=1, dilation=1)
self.bridge_bn1 = nn.BatchNorm1d(512)
self.bridge2 = nn.Conv1d(512, 512, kernel_size=3, padding=2, dilation=2)
self.bridge_bn2 = nn.BatchNorm1d(512)
self.bridge3 = nn.Conv1d(512, 512, kernel_size=3, padding=4, dilation=4)
self.bridge_bn3 = nn.BatchNorm1d(512)
```

#### Why Dilated Convolutions Instead of BiLSTM?

The original architecture used a Bidirectional LSTM to capture long-range temporal dependencies. However, this caused two critical problems:

1. **Vanishing Gradients:** The bottleneck sequence is 625 samples long. LSTMs process sequences step-by-step, and gradients must flow through all 625 steps during backpropagation. Despite the LSTM's gating mechanism, gradients tend to vanish over sequences longer than ~200 steps, causing the model to effectively stop learning from distant parts of the signal.

2. **Shared BatchNorm Bug:** In an earlier implementation, a single `BatchNorm1d(512)` instance was accidentally reused across all three convolution layers. During training (`model.train()`), BatchNorm computes statistics from the current batch, so the bug was partially masked. During evaluation (`model.eval()`), BatchNorm uses stored running statistics — but the shared instance had statistics corrupted by three different feature distributions, producing near-random outputs.

**How Dilated Convolutions Work:**

A standard Conv1d with `kernel_size=3` looks at 3 consecutive samples. A dilated Conv1d with `kernel_size=3, dilation=2` looks at samples `[t-2, t, t+2]` — skipping every other sample. With `dilation=4`, it looks at `[t-4, t, t+4]`.

**Effective receptive field per layer:**
- Layer 1 (dilation=1): 3 samples
- Layer 2 (dilation=2): 5 samples
- Layer 3 (dilation=4): 9 samples
- **Combined receptive field:** 1 + 2 + 4 + 4 + 2 + 1 = **15 samples** per position (at the 625-sample bottleneck level, which corresponds to 15 × 8 = **120 samples** in the original 5000-sample space, or **240ms** at 500Hz).

This 240ms receptive field is more than sufficient to capture full P-wave instances (which are typically 80–120ms) while maintaining perfectly stable gradient flow through standard convolutional backpropagation.

**Why separate BatchNorm instances:**
```python
self.bridge_bn1 = nn.BatchNorm1d(512)
self.bridge_bn2 = nn.BatchNorm1d(512)
self.bridge_bn3 = nn.BatchNorm1d(512)
```
Each conv layer produces features with different statistical distributions. Each BatchNorm must independently track its own running mean and variance to produce correct normalization during evaluation.

### Decoder (Spatial Reconstruction)

```python
self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)  # 625 → 1250
self.dec3 = AttentionalInception(512, 256)                          # 256+256 → 256
```

**Why ConvTranspose1d:** This is the "learnable upsampling" operation (also called transposed convolution or deconvolution). With `kernel_size=2, stride=2`, it doubles the spatial dimension, reversing the MaxPool operation from the encoder.

**Why 512 input channels to dec3:** The upsampled features (256 channels) are concatenated with the skip connection from enc3 (256 channels), producing 512 channels. This is the core U-Net principle: skip connections carry high-resolution spatial details from the encoder to the decoder, preventing the "lossy bottleneck" problem.

### Output Heads

```python
self.heatmap_head = nn.Sequential(
    nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Dropout(0.1), nn.Conv1d(32, 1, 1), nn.Sigmoid()
)
```
**Why Sigmoid:** The heatmap represents confidence probability (0 = no P-wave, 1 = P-wave center). Sigmoid squashes the output to the [0, 1] range.

**Why Dropout(0.1):** Light dropout (10%) on the heatmap prevents overconfident predictions during training, which improves generalization to unseen test data.

```python
self.width_head = nn.Sequential(
    nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 1, 1)  # No activation — raw regression output
)
```
**Why no final activation:** Width is a regression target (continuous positive value). Applying Sigmoid would cap it at 1.0, and ReLU would prevent learning from negative gradient signals. Raw linear output allows the network to predict any width value.

```python
self.mask_head = nn.Sequential(
    nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 1, 1), nn.Sigmoid()
)
```
**Why Sigmoid:** The mask is a binary classification (each sample is either inside a P-wave or not), so Sigmoid produces the required [0, 1] probability.

---

## 4.5 Ablation Model: `AtrionNetBaseline`

```python
class AtrionNetBaseline(nn.Module):
    def __init__(self, in_channels=12):
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, 64, 9, padding=4), nn.ReLU())
        self.pool = nn.MaxPool1d(2)
        self.dec = nn.Sequential(nn.ConvTranspose1d(64, 64, 2, 2), nn.Conv1d(64, 64, 3, padding=1))
```

**Purpose:** This is a deliberately minimal "vanilla CNN" model used for **ablation study** — a scientific comparison to prove that the Attentional Inception blocks and Dilated Bottleneck are actually necessary. If this baseline achieves similar performance to the Hybrid model, it would mean the complex architecture provides no benefit. If it performs significantly worse, it validates the engineering decisions.

---

## 4.6 Total Parameter Count

| Component | Parameters (Approx.) |
|---|---|
| Encoder (3 × AttentionalInception) | ~800K |
| Dilated Bottleneck (3 × Conv1d) | ~2.4M |
| Decoder (3 × AttentionalInception + 3 × ConvTranspose1d) | ~1.2M |
| Output Heads (3 × small CNN) | ~10K |
| **Total** | **~4.4M** |

This is relatively lightweight for a deep learning model, making it suitable for training on a small dataset (200 LUDB records) without severe overfitting.

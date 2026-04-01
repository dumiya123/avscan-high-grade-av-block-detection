"""
1D U-Net Baseline Model
=======================
A faithful 1D implementation of the classic U-Net architecture (Ronneberger et al., 2015),
adapted for temporal ECG signals. This is the most common deep learning baseline used
in ECG segmentation literature, including the Joung et al. (2024) paper.

Key difference from AtrionNet:
- This model is a pure SEGMENTATION model (pixel-wise binary mask)
- It does NOT produce instance-level detections (no heatmap, no width, no NMS)
- It CANNOT handle overlapping waveforms — it outputs one label per time sample
- It assigns the label "P-wave" or "not P-wave" to each of the 5000 samples

This is being used as a baseline to prove that segmentation is insufficient for the
overlapping P-wave problem in High-Grade AV Block.
"""

import torch
import torch.nn as nn


class DoubleConv1D(nn.Module):
    """Two consecutive Conv1d → BatchNorm → ReLU blocks"""
    def __init__(self, in_ch, out_ch, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D_Baseline(nn.Module):
    """
    Standard 1D U-Net with 4 encoder levels and 4 decoder levels.
    Input:  [B, 12, 5000]  — 12-lead ECG
    Output: dict with 'mask' [B, 1, 5000] — binary segmentation mask

    NOTE: Unlike AtrionNet, there is NO heatmap head and NO width head.
    P-wave instances must be extracted by finding contiguous segments where
    the mask > 0.5. This has a fundamental limitation: if two P-waves overlap,
    they merge into a single segment and cannot be separated.
    """

    def __init__(self, in_channels=12, features=[64, 128, 256, 512]):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv1D(ch, f))
            self.pools.append(nn.MaxPool1d(2))
            ch = f

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_features = list(reversed(features))
        in_ch = features[-1] * 2
        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose1d(in_ch, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv1D(f * 2, f))
            in_ch = f

        # ── Output Head ──────────────────────────────────────────────────────
        # Binary segmentation: each sample is "inside a P-wave" or not
        self.mask_head = nn.Sequential(
            nn.Conv1d(features[0], 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Dummy heatmap/width heads to be compatible with the evaluation pipeline
        self.heatmap_head = nn.Sequential(nn.Conv1d(features[0], 1, 1), nn.Sigmoid())
        self.width_head   = nn.Sequential(nn.Conv1d(features[0], 1, 1))

    def forward(self, x):
        # Encode
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decode
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle odd-length mismatch
            if x.shape[-1] != skip.shape[-1]:
                x = torch.nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = decoder(torch.cat([skip, x], dim=1))

        return {
            'heatmap': self.heatmap_head(x),  # Used downstream for instance extraction
            'width':   self.width_head(x),
            'mask':    self.mask_head(x),
        }


if __name__ == '__main__':
    model = UNet1D_Baseline(in_channels=12)
    x = torch.randn(2, 12, 5000)
    out = model(x)
    print(f"[UNet1D_Baseline] Mask shape: {out['mask'].shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[UNet1D] Total trainable parameters: {total_params:,}")

"""
Simple 1D CNN Baseline Model
============================
This is a deliberately simple convolutional neural network used as a
benchmarking baseline. It has NO attention mechanism, NO Inception blocks,
and NO dilated convolutions — it is a plain encoder-decoder.

Purpose: To prove that AtrionNet's architectural innovations actually contribute
to performance. If AtrionNet outperforms this, the complexity is justified.
"""

import torch
import torch.nn as nn


class SimpleCNNBaseline(nn.Module):
    """
    A plain, shallow convolutional encoder-decoder for P-wave detection.
    Architecture:
        - Encoder: 3 standard Conv1d blocks with MaxPool
        - Bottleneck: 2 plain Conv1d layers (no dilation)
        - Decoder: 3 ConvTranspose1d blocks with skip connections
        - Output: Single heatmap head only (no width or mask heads)

    This is the absolute minimum viable architecture for comparison.
    """

    def __init__(self, in_channels=12):
        super(SimpleCNNBaseline, self).__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = self._conv_block(in_channels, 64)   # [B,  64, 5000]
        self.pool1 = nn.MaxPool1d(2)                     # [B,  64, 2500]

        self.enc2 = self._conv_block(64, 128)            # [B, 128, 2500]
        self.pool2 = nn.MaxPool1d(2)                     # [B, 128, 1250]

        self.enc3 = self._conv_block(128, 256)           # [B, 256, 1250]
        self.pool3 = nn.MaxPool1d(2)                     # [B, 256,  625]

        # ── Bottleneck (plain, no dilation) ──────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3  = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.up2  = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.up1  = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        # ── Output: Heatmap only ─────────────────────────────────────────────
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Width and mask heads are absent — this baseline cannot predict them
        self.width_head = nn.Sequential(nn.Conv1d(64, 1, kernel_size=1))
        self.mask_head  = nn.Sequential(nn.Conv1d(64, 1, kernel_size=1), nn.Sigmoid())

    def _conv_block(self, in_ch, out_ch):
        """Standard double-conv block: Conv → BN → ReLU → Conv → BN → ReLU"""
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return {
            'heatmap': self.heatmap_head(d1),
            'width':   self.width_head(d1),
            'mask':    self.mask_head(d1),
        }


if __name__ == '__main__':
    model = SimpleCNNBaseline(in_channels=12)
    x = torch.randn(2, 12, 5000)
    out = model(x)
    print(f"[SimpleCNNBaseline] Heatmap shape: {out['heatmap'].shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SimpleCNNBaseline] Total trainable parameters: {total_params:,}")

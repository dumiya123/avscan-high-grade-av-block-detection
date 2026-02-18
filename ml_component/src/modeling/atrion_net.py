"""
AtrionNet: Anchor-Free 1D Instance Segmentation Model.
Optimized for detecting and quantifying dissociated P-waves in High-Grade AV block.

Architecture:
- Encoder: Inception Blocks (multi-scale feature extraction)
- Decoder: 1D Transposed Convolutions (upsampling to original resolution)
- Prediction Heads (Anchor-Free):
    1. Heatmap Head: Detects wave centers (Dissociated P-waves)
    2. Width Head: Predicts temporal duration (Instance width)
    3. Mask Head: Sample-wise segmentation for each instance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock1D, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        
        self.conv_small = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=9, padding=4)
        self.conv_medium = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=19, padding=9)
        self.conv_large = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=39, padding=19)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bottleneck(x)
        out1 = self.conv_small(x)
        out2 = self.conv_medium(x)
        out3 = self.conv_large(x)
        out4 = x # Residual path
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return self.relu(self.bn(out))

class AtrionNetSegmentation(nn.Module):
    def __init__(self, in_channels=12, num_instances=1):
        super(AtrionNetSegmentation, self).__init__()
        
        # 1. Encoder (Inception-based scale capture)
        self.enc1 = InceptionBlock1D(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = InceptionBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = InceptionBlock1D(128, 256)
        self.pool3 = nn.MaxPool1d(2)
        
        # 2. Bridge
        self.bridge = InceptionBlock1D(256, 512)
        
        # 3. Decoder
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = InceptionBlock1D(512, 256)
        
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = InceptionBlock1D(256, 128)
        
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = InceptionBlock1D(128, 64)

        # 4. Anchor-Free HEADS (The Research Innovation)
        # Heatmap branch: Probability of a P-wave center at this sample
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Width branch: Normalized temporal duration of the wave at this center
        self.width_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        # Mask branch: Sample-wise segmentation (is this sample part of the instance?)
        self.mask_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # [64, 5000]
        e2 = self.enc2(self.pool1(e1)) # [128, 2500]
        e3 = self.enc3(self.pool2(e2)) # [256, 1250]
        
        # Bridge
        b = self.bridge(self.pool3(e3)) # [512, 625]
        
        # Decoder with Skip Connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Anchor-Free Outputs
        heatmap = self.heatmap_head(d1)
        width = self.width_head(d1)
        mask = self.mask_head(d1)
        
        return {
            'heatmap': heatmap,
            'width': width,
            'mask': mask
        }

if __name__ == "__main__":
    # Test the model
    model = AtrionNetSegmentation(in_channels=12)
    dummy_input = torch.randn(1, 12, 5000)
    output = model(dummy_input)
    print(f"Heatmap shape: {output['heatmap'].shape}")
    print(f"Width shape: {output['width'].shape}")
    print(f"Mask shape: {output['mask'].shape}")

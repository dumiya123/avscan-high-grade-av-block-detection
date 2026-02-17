
"""
AtrionNet Model Architecture: Multi-Task Attention U-Net
This module defines the primary neural network architecture used for ECG analysis.

Key Concepts:
1.  **1D Instance Segmentation**: Unlike semantic segmentation, we need to distinguish 
    individual waves (instances) accurately.
2.  **Multi-Task Learning (MTL)**: Simultaneously performing segmentation (local) 
    and classification (global) to leverage shared features.
3.  **Attention Fusion**: Using spatial and channel attention to suppress noise 
    and highlight relevant diagnostic segments.
"""

import torch
import torch.nn as nn
from .attention import AttentionGate, ChannelAttention


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for 1D Signal Processing.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder (Downsampling) Stage of the U-Net.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.channel_att = ChannelAttention(out_channels)
    
    def forward(self, x):
        skip = self.conv(x)
        skip = self.channel_att(skip)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder (Upsampling) Stage with Spatial Attention Fusion.
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        
        if use_attention:
            self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        else:
            self.attention = None
        
        self.conv = ConvBlock(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        if x.shape[2] != skip.shape[2]:
            x = nn.functional.interpolate(x, size=skip.shape[2], mode='linear', align_corners=False)
        
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class ECGUNet(nn.Module):
    """
    AtrionNet Master Architecture: 1D Attention-Fused U-Net.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 5,
        num_clf_classes: int = 6,
        base_channels: int = 64,
        use_attention: bool = True
    ):
        super(ECGUNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_seg_classes = num_seg_classes
        self.num_clf_classes = num_clf_classes
        
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels * 16, base_channels * 16, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, use_attention)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, use_attention)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, use_attention)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, use_attention)
        
        self.seg_head = nn.Conv1d(base_channels, num_seg_classes, kernel_size=1)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.clf_head = nn.Sequential(
            nn.Linear(base_channels * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_clf_classes)
        )
    
    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        bottleneck_features = self.bottleneck(x)
        
        clf_features = self.global_pool(bottleneck_features).squeeze(-1)
        clf_out = self.clf_head(clf_features)
        
        x = self.dec4(bottleneck_features, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        seg_out = self.seg_head(x)
        
        return seg_out, clf_out
    
    def get_attention_maps(self, x):
        # Simplified implementation for saving space
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        bottleneck_features = self.bottleneck(x)
        
        maps = {}
        # Simple extraction logic could go here
        return maps


def model_summary(model):
    """
    Print model architecture summary and parameter count.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "=" * 60)
    print("ATRION-NET MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Type:                Attention U-Net (1D)")
    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Params:    {trainable_params:,}")
    print(f"Non-Trainable:       {total_params - trainable_params:,}")
    print("-" * 60)
    print("Encoder Depth:       4 Levels")
    print("Attention Gates:     Enabled")
    print("Multi-Task Heads:    Segmentation (5 class), Classification (6 class)")
    print("=" * 60 + "\n")


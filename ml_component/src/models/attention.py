"""
AtrionNet Attention Mechanisms: Spatial and Channel-Wise Focus
This module contains the mathematical implementation of the attention gates 
that allow AtrionNet to ignore noise and focus on diagnostic wave segments.
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Spatial Attention Gate (AG) for U-Net Skip Connections.
    
    Why it's used:
    In a standard U-Net, skip connections bring low-level noise from the encoder 
    directly to the decoder. AGs filter these connections.
    
    How it works:
    1.  It takes two inputs: the gated signal 'g' (from the deeper decoder layer) 
        and the skip signal 'x' (from the encoder).
    2.  'g' contains high-level rhythmic information.
    3.  'x' contains high-resolution spatial details.
    4.  Logic: `psi = sigmoid(relu(W_g(g) + W_x(x)))`. 
        This generates a map from 0 to 1. Areas of the ECG with waves get 
        near-1 values (preserved), while background noise gets near-0 (suppressed).
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        # Transform skip signal
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        # Attention coefficient generator
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # Upsample gating signal to match resolution of skip signal
        if g.shape[2] != x.shape[2]:
            g = nn.functional.interpolate(g, size=x.shape[2], mode='linear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Element-wise addition of features causes 
        # overlapping activations to strengthen (waves)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Multiply skip connection by the attention map
        return x * psi


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Style Channel Attention.
    
    Why it's used:
    Not all convolutional filters are equally useful. Some filters might reliably 
    detect P-waves, while others might just pick up baseline drift.
    
    How it works:
    1.  **Squeeze**: Global Average/Max Pooling collapses the ECG sequence 
        into a single vector representing the 'energy' of each filter.
    2.  **Excitation**: A small two-layer neural net (Bottleneck) learns which 
        channels should be 'excited' (amplified) and which 'inhibited'.
    3.  **Scale**: Multiplies the original signal by the learned channel weights.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Shared MLP for learning channel dependencies
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, C, _ = x.shape
        
        # Gather global statistics for each channel
        avg_out = self.fc(self.avg_pool(x).view(batch_size, C))
        max_out = self.fc(self.max_pool(x).view(batch_size, C))
        
        # Generate importance score per channel
        out = self.sigmoid(avg_out + max_out).view(batch_size, C, 1)
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Channel + Spatial)
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction factor
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
            
        Returns:
            Attended features (batch, channels, seq_len)
        """
        # Channel attention
        x = self.channel_attention(x)
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_conv(spatial_input)
        
        return x * spatial_attention


if __name__ == "__main__":
    # Test attention modules
    batch_size = 4
    channels = 64
    seq_len = 1000
    
    x = torch.randn(batch_size, channels, seq_len)
    
    # Test Attention Gate
    g = torch.randn(batch_size, 128, seq_len // 2)
    att_gate = AttentionGate(F_g=128, F_l=64, F_int=32)
    out = att_gate(g, x)
    print(f"Attention Gate output: {out.shape}")
    
    # Test Self Attention
    self_att = SelfAttention(channels)
    out = self_att(x)
    print(f"Self Attention output: {out.shape}")
    
    # Test Channel Attention
    ch_att = ChannelAttention(channels)
    out = ch_att(x)
    print(f"Channel Attention output: {out.shape}")
    
    # Test CBAM
    cbam = CBAM(channels)
    out = cbam(x)
    print(f"CBAM output: {out.shape}")

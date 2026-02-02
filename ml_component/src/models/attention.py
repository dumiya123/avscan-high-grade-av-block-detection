"""
Attention mechanisms for ECG U-Net
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections in U-Net
    Helps the model focus on relevant features
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature maps in gating signal (from decoder)
            F_l: Number of feature maps in skip connection (from encoder)
            F_int: Number of intermediate feature maps
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (batch, F_g, seq_len_g)
            x: Skip connection from encoder (batch, F_l, seq_len_x)
            
        Returns:
            Attention-weighted features (batch, F_l, seq_len_x)
        """
        # Upsample gating signal if needed
        if g.shape[2] != x.shape[2]:
            g = nn.functional.interpolate(g, size=x.shape[2], mode='linear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi


class SelfAttention(nn.Module):
    """
    Self-attention module for capturing long-range dependencies
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction factor
        """
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
            
        Returns:
            Self-attended features (batch, channels, seq_len)
        """
        batch_size, C, seq_len = x.shape
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, seq_len).permute(0, 2, 1)  # (B, seq_len, C')
        key = self.key(x).view(batch_size, -1, seq_len)  # (B, C', seq_len)
        value = self.value(x).view(batch_size, -1, seq_len)  # (B, C, seq_len)
        
        # Attention map
        attention = torch.bmm(query, key)  # (B, seq_len, seq_len)
        attention = self.softmax(attention)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, seq_len)
        out = self.gamma * out + x
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention module (Squeeze-and-Excitation)
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction factor
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
            
        Returns:
            Channel-attended features (batch, channels, seq_len)
        """
        batch_size, C, _ = x.shape
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(batch_size, C))
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(batch_size, C))
        
        # Combine
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

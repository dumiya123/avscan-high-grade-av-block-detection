"""
ECG U-Net with Attention for 5-class segmentation and AV block classification
"""

import torch
import torch.nn as nn
from .attention import AttentionGate, ChannelAttention


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    
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
    """Encoder block with convolution and downsampling"""
    
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
    """Decoder block with upsampling and skip connection"""
    
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
        
        # Match dimensions
        if x.shape[2] != skip.shape[2]:
            x = nn.functional.interpolate(x, size=skip.shape[2], mode='linear', align_corners=False)
        
        # Apply attention to skip connection
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class ECGUNet(nn.Module):
    """
    1D U-Net with Attention for ECG Analysis
    
    Multi-task architecture:
    - Segmentation: 5-class pixel-wise classification
    - Classification: 6-class AV block detection
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 5,
        num_clf_classes: int = 6,
        base_channels: int = 64,
        use_attention: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (1 for single-lead ECG)
            num_seg_classes: Number of segmentation classes (5)
            num_clf_classes: Number of classification classes (6 AV block types)
            base_channels: Base number of channels
            use_attention: Whether to use attention mechanisms
        """
        super(ECGUNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_seg_classes = num_seg_classes
        self.num_clf_classes = num_clf_classes
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels * 16, base_channels * 16, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Decoder
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, use_attention)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, use_attention)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, use_attention)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, use_attention)
        
        # Segmentation head
        self.seg_head = nn.Conv1d(base_channels, num_seg_classes, kernel_size=1)
        
        # Classification head
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
        """
        Forward pass
        
        Args:
            x: Input ECG signal (batch, 1, seq_len)
            
        Returns:
            seg_out: Segmentation output (batch, num_seg_classes, seq_len)
            clf_out: Classification output (batch, num_clf_classes)
        """
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        bottleneck_features = self.bottleneck(x)
        
        # Classification branch (from bottleneck)
        clf_features = self.global_pool(bottleneck_features).squeeze(-1)
        clf_out = self.clf_head(clf_features)
        
        # Decoder
        x = self.dec4(bottleneck_features, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Segmentation output
        seg_out = self.seg_head(x)
        
        return seg_out, clf_out
    
    def get_attention_maps(self, x):
        """
        Extract attention maps for visualization
        
        Args:
            x: Input ECG signal
            
        Returns:
            Dictionary of attention maps from each decoder block
        """
        attention_maps = {}
        
        # Forward pass through encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        bottleneck_features = self.bottleneck(x)
        
        # Decoder with attention extraction
        if hasattr(self.dec4, 'attention') and self.dec4.attention is not None:
            x_up = self.dec4.upsample(bottleneck_features)
            if x_up.shape[2] != skip4.shape[2]:
                x_up = nn.functional.interpolate(x_up, size=skip4.shape[2], mode='linear', align_corners=False)
            
            # Get attention weights
            g1 = self.dec4.attention.W_g(x_up)
            x1 = self.dec4.attention.W_x(skip4)
            psi = self.dec4.attention.relu(g1 + x1)
            attention_maps['dec4'] = self.dec4.attention.psi(psi)
        
        return attention_maps


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(1, 1, 5000)):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, seq_len)
    """
    print("=" * 80)
    print("ECG U-Net Model Summary")
    print("=" * 80)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test forward pass
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)
    
    with torch.no_grad():
        seg_out, clf_out = model(x)
    
    print(f"\nInput shape: {tuple(x.shape)}")
    print(f"Segmentation output shape: {tuple(seg_out.shape)}")
    print(f"Classification output shape: {tuple(clf_out.shape)}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test model
    print("Testing ECG U-Net...")
    
    model = ECGUNet(
        in_channels=1,
        num_seg_classes=5,
        num_clf_classes=6,
        base_channels=64,
        use_attention=True
    )
    
    model_summary(model)
    
    # Test forward pass
    batch_size = 4
    seq_len = 5000
    x = torch.randn(batch_size, 1, seq_len)
    
    seg_out, clf_out = model(x)
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Segmentation: {seg_out.shape}")
    print(f"   Classification: {clf_out.shape}")
    
    # Test attention extraction
    attention_maps = model.get_attention_maps(x)
    print(f"\n✅ Extracted {len(attention_maps)} attention maps")

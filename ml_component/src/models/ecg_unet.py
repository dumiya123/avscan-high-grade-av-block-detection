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
    
    Why this matters:
    The 1D Convolution (`nn.Conv1d`) learns to identify temporal patterns like 
    sharp peaks (QRS) or slow waves (T). BatchNorm stabilizes training by 
    normalizing layer activations, and ReLU adds the non-linearity needed to 
    learn complex mappings.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            # First conv captures local morphology (e.g., slopes)
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv refines these features into higher-level wave descriptors
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder (Downsampling) Stage of the U-Net.
    
    Logic:
    1.  Convolutions extract features.
    2.  `ChannelAttention` focuses on the most 'important' filters (e.g., those detecting P-waves).
    3.  `MaxPool1d` reduces resolution by 2x, allowing the model to 'see' a wider 
        temporal context (temporal receptive field) in the next layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Squeeze-and-Excitation style attention for channel priority
        self.channel_att = ChannelAttention(out_channels)
    
    def forward(self, x):
        skip = self.conv(x)
        skip = self.channel_att(skip)
        x = self.pool(skip)
        # We return 'skip' to be fused with the decoder later
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder (Upsampling) Stage with Spatial Attention Fusion.
    
    Logic:
    This block reconstructs the signal resolution. Crucially, it uses an 
    `AttentionGate` to decide which parts of the 'skip connection' are 
    actually relevant to the current upsampling step, effectively 
    filtering out background noise.
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super(DecoderBlock, self).__init__()
        
        # Upsample temporally
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        
        if use_attention:
            # Spatial Attention: Highlight wave regions based on gated context
            self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        else:
            self.attention = None
        
        # Merge upsampled features + skip features
        self.conv = ConvBlock(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Defensive programming: ensure dimensions match after upsampling
        if x.shape[2] != skip.shape[2]:
            x = nn.functional.interpolate(x, size=skip.shape[2], mode='linear', align_corners=False)
        
        # Gated Skip Connection: Combine low-level spatial data with high-level diagnostic data
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        # Concatenate along channel dimension to fuse information
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class ECGUNet(nn.Module):
    """
    AtrionNet Master Architecture: 1D Attention-Fused U-Net.
    
    The model architecture is split into three main logical parts:
    1.  **Shared Encoder**: Learns universal features of ECG morphology.
    2.  **Segmentation Branch**: Pixel-wise wave boundary detector (Local Task).
    3.  **Classification Branch**: Global rhythm diagnosis (Global Task).
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
        
        # 1. Encoder Stack: Extracts hierarchical features (low-level to high-level)
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        # 2. Bottleneck: The model's "Compressed Knowledge" center
        # We use Dilated Convolutions here to expand the receptive field 
        # (allowing the model to 'see' multiple heartbeats) without losing resolution.
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels * 16, base_channels * 16, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 3. Decoder Stack: Reconstructs waves from abstract features
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, use_attention)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, use_attention)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, use_attention)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, use_attention)
        
        # 4. Final Segmentation Head: Maps features to 5 wave classes
        self.seg_head = nn.Conv1d(base_channels, num_seg_classes, kernel_size=1)
        
        # 5. Classification Head: Fuses global temporal data into a diagnosis
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.clf_head = nn.Sequential(
            nn.Linear(base_channels * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Prevents overfitting by randomly silencing neurons
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_clf_classes)
        )
    
    def forward(self, x):
        """
        Input: Batch of ECGs [B, 1, 5000]
        Execution: Encoder -> Bottleneck -> (Split) -> Classification Header & Decoder Header
        """
        # Feature Extraction Stage
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Abstract Knowledge Stage
        bottleneck_features = self.bottleneck(x)
        
        # TASK 1: Global Classification (Diagnosis)
        # Pull global features from the bottleneck
        clf_features = self.global_pool(bottleneck_features).squeeze(-1)
        clf_out = self.clf_head(clf_features)
        
        # TASK 2: Local Segmentation (Wave Maps)
        # Reconstruct spatial maps using skip connections
        x = self.dec4(bottleneck_features, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Map back to 5-class wave probabilities
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

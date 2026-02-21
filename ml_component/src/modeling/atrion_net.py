"""
AtrionNet Hybrid: CNN + Temporal Modeling (BiLSTM).
Optimized for high-precision P-wave quantification and dissociated P-wave detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock1D, self).__init__()
        self.out_channels = out_channels
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
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

class AtrionNetHybrid(nn.Module):
    """
    RESEARCH-GRADE HYBRID ARCHITECTURE (CNN + BiLSTM)
    Addresses the gap in temporal modeling for dissociated P-waves.
    """
    def __init__(self, in_channels=12, hidden_dim=256):
        super(AtrionNetHybrid, self).__init__()
        
        # 1. CNN Encoder (Spatial/Local Feature Extraction)
        self.enc1 = InceptionBlock1D(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = InceptionBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = InceptionBlock1D(128, 256)
        self.pool3 = nn.MaxPool1d(2)
        
        # 2. Bridge: Temporal Modeling (BiLSTM)
        # Sequence length here is L/8 = 625
        self.lstm = nn.LSTM(input_size=256, 
                           hidden_size=hidden_dim, 
                           num_layers=1, 
                           batch_first=True, 
                           bidirectional=True)
        # Projection back to 512 for decoder compatibility
        self.bridge_proj = nn.Conv1d(hidden_dim * 2, 512, kernel_size=1)
        
        # 3. Decoder with Skip Connections
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = InceptionBlock1D(512, 256)
        
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = InceptionBlock1D(256, 128)
        
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = InceptionBlock1D(128, 64)

        # 4. Anchor-Free HEADS
        # Heatmap branch: Probability of a P-wave center
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Width branch: Normalized temporal duration (Regression)
        self.width_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        # Mask branch: Sample-wise segmentation
        self.mask_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # [64, 5000]
        e2 = self.enc2(self.pool1(e1)) # [128, 2500]
        e3 = self.enc3(self.pool2(e2)) # [256, 1250]
        
        # Bridge (BiLSTM)
        # Expected dim for LSTM: [Batch, Seq, Features]
        b_in = self.pool3(e3).transpose(1, 2) # [Batch, 625, 256]
        b_out, _ = self.lstm(b_in) # [Batch, 625, 512]
        b_out = b_out.transpose(1, 2) # [Batch, 512, 625]
        b = self.bridge_proj(b_out)
        
        # Decoder with Skip Connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Outputs
        return {
            'heatmap': self.heatmap_head(d1),
            'width': self.width_head(d1),
            'mask': self.mask_head(d1)
        }

class AtrionNetBaseline(nn.Module):
    """
    Original CNN-only architecture for ablation comparison.
    """
    def __init__(self, in_channels=12):
        super(AtrionNetBaseline, self).__init__()
        self.enc1 = InceptionBlock1D(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = InceptionBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = InceptionBlock1D(128, 256)
        self.pool3 = nn.MaxPool1d(2)
        self.bridge = InceptionBlock1D(256, 512)
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = InceptionBlock1D(512, 256)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = InceptionBlock1D(256, 128)
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = InceptionBlock1D(128, 64)
        self.heatmap_head = nn.Sequential(nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 1, 1), nn.Sigmoid())
        self.width_head = nn.Sequential(nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 1, 1))
        self.mask_head = nn.Sequential(nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bridge(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return {'heatmap': self.heatmap_head(d1), 'width': self.width_head(d1), 'mask': self.mask_head(d1)}

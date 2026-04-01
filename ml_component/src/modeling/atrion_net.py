"""
Attentional-AtrionNet v4.0: Hybrid CNN + BiLSTM + Self-Attention.
Optimized for overlapping P-wave detection and high-precision quantification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock1D(nn.Module):
    """Squeeze-and-Excitation variant for 1D ECG signals."""
    def __init__(self, channels, reduction=16):
        super(AttentionBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class AttentionalInception(nn.Module):
    """Inception block with integrated Self-Attention to focus on weak P-waves."""
    def __init__(self, in_channels, out_channels):
        super(AttentionalInception, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        
        self.conv_small = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=9, padding=4)
        self.conv_medium = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=19, padding=9)
        self.conv_large = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=39, padding=19)
        
        self.attention = AttentionBlock1D(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bottleneck(x)
        out1 = self.conv_small(x)
        out2 = self.conv_medium(x)
        out3 = self.conv_large(x)
        out4 = x # Residual
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.attention(out) # Research Innovation: Attention Gating
        return self.relu(self.bn(out))

class AtrionNetHybrid(nn.Module):
    """
    V5.0 RESEARCH ARCHITECTURE: 
    Multi-Scale CNN + Dilated Convolutional Context + Attentional Gating.
    Replaced the unstable 625-step BiLSTM with a highly stable Dilated CNN Bottleneck.
    """
    def __init__(self, in_channels=12, hidden_dim=256):
        super(AtrionNetHybrid, self).__init__()
        
        # 1. Attentional Encoder
        self.enc1 = AttentionalInception(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = AttentionalInception(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = AttentionalInception(128, 256)
        self.pool3 = nn.MaxPool1d(2)
        
        # 2. Bridge: Dilated Convolutional Context (Solves BiLSTM Vanishing Gradients)
        # Sequence length: 625 samples. Dilations [1, 2, 4, 8] cover massive receptive fields.
        self.bridge1 = nn.Conv1d(256, 512, kernel_size=3, padding=1, dilation=1)
        self.bridge_bn1 = nn.BatchNorm1d(512)
        self.bridge2 = nn.Conv1d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.bridge_bn2 = nn.BatchNorm1d(512)
        self.bridge3 = nn.Conv1d(512, 512, kernel_size=3, padding=4, dilation=4)
        self.bridge_bn3 = nn.BatchNorm1d(512)
        self.bridge_relu = nn.ReLU()
        
        # 3. Decoder
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = AttentionalInception(512, 256)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = AttentionalInception(256, 128)
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = AttentionalInception(128, 64)

        # 4. Refined Output Heads
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.1), nn.Conv1d(32, 1, 1), nn.Sigmoid()
        )
        self.width_head = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        self.mask_head = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Dilated CNN Context (Stable over long sequences)
        b = self.pool3(e3)
        b = self.bridge_relu(self.bridge_bn1(self.bridge1(b)))
        b = self.bridge_relu(self.bridge_bn2(self.bridge2(b)))
        b = self.bridge_relu(self.bridge_bn3(self.bridge3(b)))
        
        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return {
            'heatmap': self.heatmap_head(d1),
            'width': self.width_head(d1),
            'mask': self.mask_head(d1)
        }

class AtrionNetBaseline(nn.Module):
    """Vanilla CNN for ablation study."""
    def __init__(self, in_channels=12):
        super(AtrionNetBaseline, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, 64, 9, padding=4), nn.ReLU())
        self.pool = nn.MaxPool1d(2)
        self.dec = nn.Sequential(nn.ConvTranspose1d(64, 64, 2, 2), nn.Conv1d(64, 64, 3, padding=1))
        self.h = nn.Sequential(nn.Conv1d(64, 1, 1), nn.Sigmoid())
        self.w = nn.Conv1d(64, 1, 1)
        self.m = nn.Sequential(nn.Conv1d(64, 1, 1), nn.Sigmoid())
    def forward(self, x):
        f = self.pool(self.enc1(x))
        d = self.dec(f)
        return {'heatmap': self.h(d), 'width': self.w(d), 'mask': self.m(d)}

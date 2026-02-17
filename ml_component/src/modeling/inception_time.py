"""
InceptionTime for 1D ECG signals.

Reference: Fawaz et al. (2020) - InceptionTime: Finding AlexNet for Time Series Classification.
Adapted from 2D Inception to 1D for multi-lead ECG classification.

Key idea: parallel convolutions with different kernel sizes capture patterns
at multiple temporal scales simultaneously (short P-waves vs wide T-waves).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock1d(nn.Module):
    """
    Single Inception block with parallel convolutions of different kernel sizes.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes: tuple = (9, 19, 39),
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck

        # Bottleneck 1x1 conv to reduce channel dim before expensive convolutions
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in_channels = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in_channels = in_channels

        # Parallel convolutions with different kernel sizes
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            pad = ks // 2
            self.convs.append(
                nn.Conv1d(conv_in_channels, num_filters, kernel_size=ks, padding=pad, bias=False)
            )

        # MaxPool branch (captures sharp features like QRS peaks)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_channels, num_filters, kernel_size=1, bias=False)

        # Total output channels = num_filters * (len(kernel_sizes) + 1)
        total_filters = num_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_filters)

    def forward(self, x):
        # Bottleneck
        if self.bottleneck is not None:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x

        # Parallel convolutions
        conv_outputs = [conv(x_bottleneck) for conv in self.convs]

        # MaxPool branch (operates on original input, not bottleneck)
        mp_out = self.mp_conv(self.maxpool(x))
        conv_outputs.append(mp_out)

        # Concatenate along channel dimension
        out = torch.cat(conv_outputs, dim=1)
        out = self.bn(out)
        out = F.relu(out)

        return out


class InceptionResidualBlock(nn.Module):
    """
    Two Inception blocks with a residual connection.
    """

    def __init__(self, in_channels, num_filters=32, bottleneck_channels=32):
        super().__init__()

        # Each InceptionBlock outputs num_filters * 4 channels
        out_channels = num_filters * 4

        self.inception1 = InceptionBlock1d(in_channels, num_filters, bottleneck_channels)
        self.inception2 = InceptionBlock1d(out_channels, num_filters, bottleneck_channels)

        # Residual shortcut
        self.use_residual = (in_channels != out_channels)
        if self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.inception1(x)
        out = self.inception2(out)
        out = out + residual
        out = F.relu(out)
        return out


class InceptionTime(nn.Module):
    """
    Full InceptionTime model for 12-lead ECG multi-label classification.

    Architecture:
    - Stack of InceptionResidualBlocks
    - Global Average Pooling
    - Fully connected head (multi-label sigmoid output)
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        num_blocks: int = 6,
        num_filters: int = 32,
        bottleneck_channels: int = 32,
        use_residual: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_blocks):
            if use_residual and i % 2 == 1:
                # Every second block is a residual block
                self.blocks.append(
                    InceptionResidualBlock(current_channels, num_filters, bottleneck_channels)
                )
            else:
                self.blocks.append(
                    InceptionBlock1d(current_channels, num_filters, bottleneck_channels)
                )
            current_channels = num_filters * 4  # output of each block

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 12, seq_len) input ECG tensor

        Returns:
            logits: (batch, num_classes) raw logits (apply sigmoid externally)
        """
        for block in self.blocks:
            x = block(x)

        x = self.gap(x).squeeze(-1)  # (batch, channels)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

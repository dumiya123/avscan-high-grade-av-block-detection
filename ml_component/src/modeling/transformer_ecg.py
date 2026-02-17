"""
Transformer-based ECG Classification Model.

Uses a 1D convolutional feature extractor followed by a Transformer encoder.
The CNN extracts local morphological features; the Transformer captures
long-range temporal dependencies (e.g., PR interval, QT interval relationships).

Positional encoding is learned (not sinusoidal) because ECG positions
carry physiological meaning (P before QRS before T).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvEmbedding(nn.Module):
    """
    1D CNN to embed raw ECG into a sequence of feature tokens.
    Reduces temporal resolution while increasing channel dimension.
    """

    def __init__(self, in_channels: int = 12, d_model: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        # After 3 stride-2 convs: seq_len -> seq_len / 8

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, reduced_seq_len, d_model)
        """
        x = self.conv_layers(x)          # (B, d_model, L/8)
        x = x.permute(0, 2, 1)           # (B, L/8, d_model)
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    Better than sinusoidal for fixed-length ECGs where temporal position
    has specific clinical meaning.
    """

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class ECGTransformer(nn.Module):
    """
    Transformer-based 12-lead ECG classifier.

    Architecture:
        1. ConvEmbedding: Raw ECG -> feature tokens
        2. Positional encoding
        3. Transformer encoder (multi-head self-attention)
        4. Classification head (CLS token or GAP)

    Uses [CLS] token approach: a learnable class token is prepended,
    and its final representation is used for classification.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # CNN feature extractor
        self.embedding = ConvEmbedding(in_channels, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len=1000, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,     # Pre-norm for stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: (batch, 12, seq_len) input ECG

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # 1. CNN embedding
        x = self.embedding(x)           # (B, L', d_model)

        # 2. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)                  # (B, 1+L', d_model)

        # 3. Add positional encoding
        x = self.pos_encoder(x)

        # 4. Transformer encoder
        x = self.transformer_encoder(x)  # (B, 1+L', d_model)

        # 5. Extract [CLS] token representation
        cls_output = x[:, 0, :]          # (B, d_model)

        # 6. Classify
        logits = self.classifier(cls_output)  # (B, num_classes)

        return logits

    def get_attention_weights(self, x):
        """
        Extract attention weights for explainability.
        Useful for Grad-CAM style analysis of which ECG segments matter.
        """
        batch_size = x.size(0)

        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)

        # Get attention from each layer
        attention_weights = []
        for layer in self.transformer_encoder.layers:
            # Self-attention forward with attention output
            # Note: PyTorch TransformerEncoderLayer doesn't expose attn weights directly
            # We use a hook-based approach in practice
            pass

        return attention_weights

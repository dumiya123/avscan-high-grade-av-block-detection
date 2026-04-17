import torch
import torch.nn as nn

class CNN1DBaseline(nn.Module):
    """Simple 1D CNN Baseline for comparison."""
    def __init__(self, in_channels=12):
        super(CNN1DBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.width_head = nn.Conv1d(64, 1, kernel_size=1)
        self.mask_head = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        feat = self.features(x)
        return {
            'heatmap': self.heatmap_head(feat),
            'width': self.width_head(feat),
            'mask': self.mask_head(feat)
        }

class CNNLSTMBaseline(nn.Module):
    """CNN-LSTM Baseline for sequence modeling without Attention."""
    def __init__(self, in_channels=12):
        super(CNNLSTMBaseline, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.width_head = nn.Conv1d(128, 1, kernel_size=1)
        self.mask_head = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, x):
        feat = self.cnn(x)
        # Permute for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        feat_lstm = feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat_lstm)
        # Permute back: (batch, seq, channels) -> (batch, channels, seq)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        return {
            'heatmap': self.heatmap_head(lstm_out),
            'width': self.width_head(lstm_out),
            'mask': self.mask_head(lstm_out)
        }

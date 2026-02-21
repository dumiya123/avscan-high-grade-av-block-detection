"""
AtrionNet Instance Dataset.
Handles loading, normalization, and research-grade augmentation for ECG signals.
"""
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class AtrionInstanceDataset(Dataset):
    def __init__(self, signals, annotations, seq_len=5000, is_train=False):
        """
        Args:
            signals: Numpy array [N, 12, 5000]
            annotations: List of dicts
            is_train: If True, applies research-grade augmentations
        """
        self.signals = signals
        self.annotations = annotations
        self.seq_len = seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.signals)

    def _normalize(self, sig):
        """Per-lead Z-score normalization."""
        mu = np.mean(sig, axis=1, keepdims=True)
        sigma = np.std(sig, axis=1, keepdims=True) + 1e-6
        return (sig - mu) / sigma

    def _augment(self, sig, centers, spans):
        """On-the-fly Research Augmentations."""
        # 1. Random Time Shift (+/- 250 samples)
        shift = np.random.randint(-250, 250)
        
        # 2. Amplitude Scaling (0.8x to 1.2x)
        scale = np.random.uniform(0.8, 1.2)
        sig = sig * scale
        
        # 3. Gaussian Noise injection
        noise = np.random.normal(0, 0.01, sig.shape)
        sig = sig + noise
        
        # 4. Lead Dropout (randomly zero 1-2 leads)
        if np.random.rand() > 0.7:
            drop_indices = np.random.choice(12, size=np.random.randint(1, 3), replace=False)
            sig[drop_indices, :] = 0
            
        # Update labels for shift
        new_centers = []
        new_spans = []
        for c in centers:
            new_c = c + shift
            if 0 <= new_c < self.seq_len:
                new_centers.append(new_c)
        for s_start, s_end in spans:
            new_s = (max(0, s_start + shift), min(self.seq_len, s_end + shift))
            new_spans.append(new_s)
            
        # Shift signal
        if shift > 0:
            sig = np.pad(sig[:, :-shift], ((0,0),(shift,0)), mode='constant')
        elif shift < 0:
            sig = np.pad(sig[:, -shift:], ((0,0),(0,-shift)), mode='constant')
            
        return sig, new_centers, new_spans

    def _generate_heatmap(self, centers, sigma=15):
        """Generates Gaussian-smoothed heatmap."""
        heatmap = np.zeros(self.seq_len)
        for center in centers:
            x = np.arange(self.seq_len)
            diff = (x - center)**2
            heatmap += np.exp(-(diff) / (2 * sigma**2))
        return np.clip(heatmap, 0, 1)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy()
        ann = self.annotations[idx]
        
        centers = [p[1] for p in ann['p_waves']]
        spans = [(p[0], p[2]) for p in ann['p_waves']]
        
        if self.is_train:
            sig, centers, spans = self._augment(sig, centers, spans)
            
        # Normalize
        sig = self._normalize(sig)
        
        # Targets
        heatmap = np.zeros((1, self.seq_len))
        width_map = np.zeros((1, self.seq_len))
        mask = np.zeros((1, self.seq_len))
        
        heatmap[0] = self._generate_heatmap(centers)
        for (s_start, s_end), center in zip(spans, centers):
            if 0 <= center < self.seq_len:
                width_map[0, int(center)] = (s_end - s_start) / self.seq_len
                mask[0, int(s_start):int(s_end)] = 1
            
        return {
            'signal': torch.FloatTensor(sig),
            'heatmap': torch.FloatTensor(heatmap),
            'width': torch.FloatTensor(width_map),
            'mask': torch.FloatTensor(mask)
        }

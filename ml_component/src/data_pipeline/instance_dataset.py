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

    def _augment(self, sig, centers, spans, idx):
        """On-the-fly Research Augmentations with Overlapping MixUp."""
        # 1. Random MixUp (30% chance) - THE RESEARCH INNOVATION
        # Overlay another real record to simulate overlapping/dissociated waves
        if np.random.rand() < 0.3:
            rand_idx = np.random.randint(0, len(self.signals))
            mix_sig = self.signals[rand_idx].copy()
            mix_ann = self.annotations[rand_idx]
            
            # Weighted addition (0.7 primary + 0.3 noise-like secondary)
            sig = (sig * 0.7) + (mix_sig * 0.3)
            
            # Combine P-wave labels
            for p in mix_ann['p_waves']:
                centers.append(p[1])
                spans.append((p[0], p[2]))

        # 2. Random Time Shift (+/- 250 samples)
        shift = np.random.randint(-250, 250)
        
        # 3. Amplitude Scaling & Noise
        sig = sig * np.random.uniform(0.8, 1.2)
        sig += np.random.normal(0, 0.01, sig.shape)
        
        # 4. Lead Dropout
        if np.random.rand() > 0.8:
            drop_indices = np.random.choice(12, size=np.random.randint(1, 3), replace=False)
            sig[drop_indices, :] = 0
            
        # Update labels for shift
        new_centers, new_spans = [], []
        for c in centers:
            new_c = c + shift
            if 0 <= new_c < self.seq_len: new_centers.append(new_c)
        for s_start, s_end in spans:
            new_s = (max(0, s_start + shift), min(self.seq_len, s_end + shift))
            new_spans.append(new_s)
            
        # Shift signal
        if shift > 0:
            sig = np.pad(sig[:, :-shift], ((0,0),(shift,0)), mode='constant')
        elif shift < 0:
            sig = np.pad(sig[:, -shift:], ((0,0),(0,-shift)), mode='constant')
            
        return sig, new_centers, new_spans

    def _generate_heatmap(self, centers, sigma=5):
        """
        Generates Tight Gaussian-smoothed heatmap.
        Sigma=5 (10ms @ 500Hz) forces High-Precision localization.
        """
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
            sig, centers, spans = self._augment(sig, centers, spans, idx)
            
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

def focal_loss(pred, target, alpha=2, beta=4):
    """Refined Focal Loss with Hard-Negative Emphasis."""
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    # Increase weight for 'Hard' negatives (like T-waves)
    neg_weights = torch.pow(1 - target, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for mask segmentation."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def create_instance_loss(pred, target):
    """AtrionNet v4.0: Balanced Loss for Overlapping Detection."""
    import torch.nn.functional as F
    
    # 1. Detection (Increased Focal weight to fix low Precision)
    hm_loss = focal_loss(pred['heatmap'], target['heatmap'])
    
    # 2. Quantification
    center_mask = target['heatmap'] > 0.95
    if center_mask.sum() > 0:
        w_loss = F.smooth_l1_loss(pred['width'][center_mask], target['width'][center_mask])
    else:
        w_loss = torch.tensor(0.0).to(pred['width'].device)
        
    # 3. Segmentation
    m_loss = F.binary_cross_entropy(pred['mask'], target['mask']) + dice_loss(pred['mask'], target['mask'])
    
    # Final Balance (Heavy weighting on Heatmap to resolve FP issues)
    return (20.0 * hm_loss) + (1.0 * w_loss) + (5.0 * m_loss)

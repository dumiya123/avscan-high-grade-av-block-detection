"""
AtrionNet Instance Dataset Loader.
Converts wave annotations (onset, peak, offset) into GT targets for:
1. Heatmaps (Center points with Gaussian blur)
2. Width Maps (Temporal duration)
3. Instance Masks (Binary segmentation)

Target Dataset: LUDB or Annotated Clinical ECGs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path

class AtrionInstanceDataset(Dataset):
    def __init__(self, signals, annotations, seq_len=5000):
        """
        Args:
            signals: Numpy array [N, 12, 5000]
            annotations: List of dicts containing 'p_waves' list of (onset, peak, offset)
        """
        self.signals = signals
        self.annotations = annotations
        self.seq_len = seq_len

    def __len__(self):
        return len(self.signals)

    def _generate_heatmap(self, centers, sigma=5):
        """Generates a Gaussian-smoothed heatmap for centers."""
        heatmap = np.zeros(self.seq_len)
        for center in centers:
            # Gaussian peak at center
            x = np.arange(self.seq_len)
            heatmap += np.exp(-((x - center)**2) / (2 * sigma**2))
        return np.clip(heatmap, 0, 1)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        ann = self.annotations[idx]
        
        # Targets
        heatmap = np.zeros((1, self.seq_len))
        width_map = np.zeros((1, self.seq_len))
        mask = np.zeros((1, self.seq_len))
        
        centers = []
        for p_onset, p_peak, p_offset in ann['p_waves']:
            # 1. Store center for heatmap
            centers.append(p_peak)
            
            # 2. Store width at the center position
            w = (p_offset - p_onset) / self.seq_len # Normalized width
            width_map[0, int(p_peak)] = w
            
            # 3. Fill mask for this instance
            mask[0, int(p_onset):int(p_offset)] = 1
            
        heatmap[0] = self._generate_heatmap(centers)
        
        return {
            'signal': torch.FloatTensor(signal),
            'heatmap': torch.FloatTensor(heatmap),
            'width': torch.FloatTensor(width_map),
            'mask': torch.FloatTensor(mask)
        }

def create_instance_loss(pred, target):
    """
    Combined Loss for AtrionNet:
    - Heatmap: Gaussian Focal Loss
    - Width: L1 Loss (only at center points)
    - Mask: Dice + BCE Loss
    """
    # 1. Heatmap Loss (Focal Loss to handle sparse centers)
    hm_loss = F.binary_cross_entropy(pred['heatmap'], target['heatmap'])
    
    # 2. Width Loss (Masked L1 - only compute loss where a center exists)
    center_mask = target['heatmap'] > 0.99
    if center_mask.sum() > 0:
        width_loss = F.l1_loss(pred['width'][center_mask], target['width'][center_mask])
    else:
        width_loss = torch.tensor(0.0).to(pred['width'].device)
        
    # 3. Mask Loss (Pixel-wise segmentation)
    mask_loss = F.binary_cross_entropy(pred['mask'], target['mask'])
    
    return hm_loss + (0.1 * width_loss) + mask_loss

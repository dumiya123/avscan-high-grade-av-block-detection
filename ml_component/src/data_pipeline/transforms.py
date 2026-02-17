"""
Data augmentation transforms for ECG signals
"""

import torch
import numpy as np
from typing import Tuple


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, signal: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            signal, seg_mask = transform(signal, seg_mask)
        return signal, seg_mask


class RandomCrop:
    """Randomly crop a segment from the ECG"""
    
    def __init__(self, crop_length: int = 4000, p: float = 0.5):
        """
        Args:
            crop_length: Length of cropped segment
            p: Probability of applying transform
        """
        self.crop_length = crop_length
        self.p = p
    
    def __call__(self, signal: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return signal, seg_mask
        
        seq_len = signal.shape[1]
        
        if seq_len <= self.crop_length:
            return signal, seg_mask
        
        # Random start position
        start = torch.randint(0, seq_len - self.crop_length, (1,)).item()
        end = start + self.crop_length
        
        # Crop and pad back to original length
        cropped_signal = signal[:, start:end]
        cropped_mask = seg_mask[start:end]
        
        # Pad to original length
        pad_length = seq_len - self.crop_length
        signal = torch.nn.functional.pad(cropped_signal, (0, pad_length), mode='constant')
        seg_mask = torch.nn.functional.pad(cropped_mask, (0, pad_length), mode='constant')
        
        return signal, seg_mask


class AddGaussianNoise:
    """Add Gaussian noise to ECG signal"""
    
    def __init__(self, snr_db: Tuple[float, float] = (15.0, 25.0), p: float = 0.5):
        """
        Args:
            snr_db: Range of SNR in dB (min, max)
            p: Probability of applying transform
        """
        self.snr_db = snr_db
        self.p = p
    
    def __call__(self, signal: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return signal, seg_mask
        
        # Random SNR
        snr_db = torch.rand(1).item() * (self.snr_db[1] - self.snr_db[0]) + self.snr_db[0]
        
        # Calculate signal power
        signal_power = torch.mean(signal ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        
        # Add noise
        noisy_signal = signal + noise
        
        return noisy_signal, seg_mask


class BaselineWander:
    """Simulate baseline wander (low-frequency drift)"""
    
    def __init__(self, freq_range: Tuple[float, float] = (0.05, 0.5), 
                 amplitude_range: Tuple[float, float] = (0.1, 0.3), 
                 p: float = 0.3):
        """
        Args:
            freq_range: Frequency range in Hz (min, max)
            amplitude_range: Amplitude range (min, max)
            p: Probability of applying transform
        """
        self.freq_range = freq_range
        self.amplitude_range = amplitude_range
        self.p = p
    
    def __call__(self, signal: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return signal, seg_mask
        
        seq_len = signal.shape[1]
        
        # Random frequency and amplitude
        freq = torch.rand(1).item() * (self.freq_range[1] - self.freq_range[0]) + self.freq_range[0]
        amplitude = torch.rand(1).item() * (self.amplitude_range[1] - self.amplitude_range[0]) + self.amplitude_range[0]
        
        # Generate baseline wander (sine wave)
        t = torch.arange(seq_len, dtype=torch.float32)
        baseline = amplitude * torch.sin(2 * np.pi * freq * t / 500)  # Assuming 500 Hz
        
        # Add to signal
        wandered_signal = signal + baseline.unsqueeze(0)
        
        return wandered_signal, seg_mask


class AmplitudeScale:
    """Random amplitude scaling"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        """
        Args:
            scale_range: Scaling factor range (min, max)
            p: Probability of applying transform
        """
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, signal: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return signal, seg_mask
        
        # Random scale factor
        scale = torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        
        # Scale signal
        scaled_signal = signal * scale
        
        return scaled_signal, seg_mask


class TimeWarp:
    """Subtle temporal distortion"""
    
    def __init__(self, warp_factor: float = 0.1, p: float = 0.3):
        """
        Args:
            warp_factor: Maximum warping factor
            p: Probability of applying transform
        """
        self.warp_factor = warp_factor
        self.p = p
    
    def __call__(self, signal: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return signal, seg_mask
        
        seq_len = signal.shape[1]
        
        # Random warp
        warp = 1.0 + (torch.rand(1).item() * 2 - 1) * self.warp_factor
        
        # New length
        new_len = int(seq_len * warp)
        
        # Resample
        warped_signal = torch.nn.functional.interpolate(
            signal.unsqueeze(0),
            size=new_len,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        # Resize back to original length
        warped_signal = torch.nn.functional.interpolate(
            warped_signal.unsqueeze(0),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        return warped_signal, seg_mask


class RandomFlip:
    """Random vertical flip (invert signal)"""
    
    def __init__(self, p: float = 0.1):
        """
        Args:
            p: Probability of applying transform
        """
        self.p = p
    
    def __call__(self, signal: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return signal, seg_mask
        
        # Flip signal vertically
        flipped_signal = -signal
        
        return flipped_signal, seg_mask


def get_train_transforms():
    """Get default training augmentation pipeline"""
    return Compose([
        AddGaussianNoise(snr_db=(15, 25), p=0.5),
        BaselineWander(p=0.3),
        AmplitudeScale(scale_range=(0.9, 1.1), p=0.5),
        TimeWarp(warp_factor=0.05, p=0.2)
    ])


def get_val_transforms():
    """Get validation transforms (no augmentation)"""
    return None  # No augmentation for validation


if __name__ == "__main__":
    # Test transforms
    signal = torch.randn(1, 5000)
    seg_mask = torch.randint(0, 5, (5000,))
    
    transforms = get_train_transforms()
    
    if transforms:
        aug_signal, aug_mask = transforms(signal, seg_mask)
        print(f"Original signal shape: {signal.shape}")
        print(f"Augmented signal shape: {aug_signal.shape}")
        print(f"Signal changed: {not torch.allclose(signal, aug_signal)}")

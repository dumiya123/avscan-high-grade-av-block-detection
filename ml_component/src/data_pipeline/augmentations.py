"""
ECG-Specific Data Augmentations.

Each augmentation is physiologically motivated:
- Time shift: simulate slight electrode placement variation
- Amplitude scaling: simulate gain differences across devices
- Gaussian noise: simulate electronic noise
- Baseline wander: simulate respiratory artifact

All transforms operate on numpy arrays of shape (num_leads, seq_len).
"""

import numpy as np
from typing import Tuple


class ECGAugmentor:
    """
    Composable ECG augmentation pipeline.
    Each augmentation is applied independently with a given probability.
    """

    def __init__(
        self,
        time_shift_max: int = 50,
        amplitude_scale_range: Tuple[float, float] = (0.8, 1.2),
        gaussian_noise_std: float = 0.05,
        baseline_wander_freq: float = 0.5,
        baseline_wander_amp: float = 0.1,
        probability: float = 0.5,
        fs: int = 500,
    ):
        self.time_shift_max = time_shift_max
        self.amp_low, self.amp_high = amplitude_scale_range
        self.noise_std = gaussian_noise_std
        self.bw_freq = baseline_wander_freq
        self.bw_amp = baseline_wander_amp
        self.p = probability
        self.fs = fs

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a 12-lead ECG signal.

        Args:
            signal: (num_leads, seq_len) numpy array

        Returns:
            Augmented signal with same shape.
        """
        if np.random.rand() < self.p:
            signal = self.time_shift(signal)

        if np.random.rand() < self.p:
            signal = self.amplitude_scale(signal)

        if np.random.rand() < self.p:
            signal = self.add_gaussian_noise(signal)

        if np.random.rand() < self.p:
            signal = self.add_baseline_wander(signal)

        return signal

    def time_shift(self, signal: np.ndarray) -> np.ndarray:
        """
        Circular shift along the time axis.
        Simulates slight misalignment in recording start.
        """
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        return np.roll(signal, shift, axis=1)

    def amplitude_scale(self, signal: np.ndarray) -> np.ndarray:
        """
        Random per-lead amplitude scaling.
        Simulates gain differences between recording devices.
        """
        num_leads = signal.shape[0]
        scales = np.random.uniform(self.amp_low, self.amp_high, size=(num_leads, 1))
        return signal * scales

    def add_gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise.
        Simulates electronic measurement noise.
        """
        noise = np.random.normal(0, self.noise_std, size=signal.shape)
        return signal + noise

    def add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """
        Add sinusoidal baseline wander.
        Simulates respiratory artifact (0.1-0.5 Hz).
        """
        num_leads, seq_len = signal.shape
        t = np.arange(seq_len) / self.fs

        # Random frequency and phase for each lead
        for lead in range(num_leads):
            freq = np.random.uniform(0.1, self.bw_freq)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0, self.bw_amp)
            wander = amp * np.sin(2 * np.pi * freq * t + phase)
            signal[lead] += wander

        return signal


class DropLeads:
    """
    Randomly zero out entire leads during training.
    Forces the model to not over-rely on any single lead.
    """

    def __init__(self, max_leads_to_drop: int = 2, probability: float = 0.2):
        self.max_drop = max_leads_to_drop
        self.p = probability

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            num_leads = signal.shape[0]
            n_drop = np.random.randint(1, self.max_drop + 1)
            drop_idx = np.random.choice(num_leads, n_drop, replace=False)
            signal[drop_idx] = 0.0
        return signal


class ComposeTransforms:
    """Chain multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            signal = t(signal)
        return signal


def get_train_augmentation(config: dict = None):
    """
    Build default training augmentation pipeline.

    Args:
        config: Augmentation config dict (from experiment.py).
                If None, uses sensible defaults.

    Returns:
        Callable transform.
    """
    if config is None:
        config = {
            'time_shift_max': 50,
            'amplitude_scale_range': (0.8, 1.2),
            'gaussian_noise_std': 0.05,
            'baseline_wander_freq': 0.5,
            'baseline_wander_amp': 0.1,
            'probability': 0.5,
        }

    return ComposeTransforms([
        ECGAugmentor(
            time_shift_max=config['time_shift_max'],
            amplitude_scale_range=config['amplitude_scale_range'],
            gaussian_noise_std=config['gaussian_noise_std'],
            baseline_wander_freq=config['baseline_wander_freq'],
            baseline_wander_amp=config['baseline_wander_amp'],
            probability=config['probability'],
        ),
        DropLeads(max_leads_to_drop=2, probability=0.15),
    ])

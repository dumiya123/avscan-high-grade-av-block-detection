import random
import math
import torch
import numpy as np

def _tnoise_powerline(fs=500, N=5000, C=1.0, fn=50.0, K=3):
    """
    Synthesize powerline noise (50Hz or 60Hz) with harmonics.
    Matched to 500Hz sampling rate and 5000 length to fit LUDB.
    """
    t = torch.arange(0, N/fs, 1./fs)
    signal = torch.zeros(N)
    phi1 = random.uniform(0, 2*math.pi)
    
    for k in range(1, K+1):
        ak = random.uniform(0, 1)
        signal += C * ak * torch.cos(2*math.pi * k * fn * t + phi1)
        
    # Scale appropriately for ECG
    return signal * 0.05 

def _tnoise_baseline_wander(fs=500, N=5000, C=1.0, fc=0.5):
    """
    Synthesize low-frequency baseline wander (e.g., patient breathing).
    """
    fdelta = fs/N
    K = max(1, int((fc/fdelta)+0.5))
    
    t = torch.arange(0, N/fs, 1./fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    
    phase_k = torch.empty(K).uniform_(0, 2*math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    
    pre_cos = 2*math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    res = (a_k * cos).sum(dim=0)
    
    # Scale appropriately for ECG
    return C * res * 0.2

class ECGCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave):
        for t in self.transforms:
            wave = t(wave)
        return wave


class GaussianNoise():
    def __init__(self, prob=0.3, scale=0.01):
        self.scale = scale
        self.prob = prob
    
    def __call__(self, wave):
        if random.random() < self.prob:
            wave += self.scale * torch.randn_like(wave)
        return wave
    

class BaselineShift():
    def __init__(self, prob=0.3, scale=0.1):
        self.prob = prob
        self.scale = scale

    def __call__(self, wave):
        if random.random() < self.prob:
            shift = torch.randn(1).item()
            wave = wave + (self.scale * shift)
        return wave


class BaselineWander():
    def __init__(self, prob=0.3, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            wander = _tnoise_baseline_wander(fs=self.freq, N=len_wave) 
            # Apply identical wander across all channels, or channel-specific.
            # Using identical here to simulate standard respiratory movement.
            wander = wander.repeat(channels, 1)
            wave = wave + wander
        return wave


class PowerlineNoise():
    def __init__(self, prob=0.3, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            # Generates identical powerline noise across channels
            noise = _tnoise_powerline(fs=self.freq, N=len_wave)
            noise = noise.repeat(channels, 1)
            wave = wave + noise
        return wave


def get_research_augmentations():
    """Returns the validated augmentations from Joung et al. 2024"""
    return ECGCompose([
        BaselineWander(prob=0.3, freq=500),
        PowerlineNoise(prob=0.3, freq=500),
        GaussianNoise(prob=0.3, scale=0.01),
        BaselineShift(prob=0.3, scale=0.05),
    ])

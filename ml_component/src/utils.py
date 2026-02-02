"""
Utility functions for the AV Block Detection System
"""

import random
import numpy as np
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (CUDA GPU or CPU)
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    else:
        print("[WARNING] GPU not available, using CPU")
        return torch.device('cpu')


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: Dict[str, float],
    filepath: Path,
    is_best: bool = False
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = filepath.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"💾 Saved best model to {best_path}")


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load model to
        
    Returns:
        Dictionary with checkpoint information
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint


def setup_logging(log_file: Optional[Path] = None, level=logging.INFO):
    """
    Configure logging
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def plot_ecg(
    signal: np.ndarray,
    sampling_rate: int = 500,
    annotations: Optional[Dict[str, np.ndarray]] = None,
    title: str = "ECG Signal",
    figsize: tuple = (15, 4)
) -> plt.Figure:
    """
    Plot ECG signal with optional annotations
    
    Args:
        signal: ECG signal array
        sampling_rate: Sampling rate in Hz
        annotations: Dictionary of annotations (e.g., {'P': [100, 500], 'QRS': [200, 600]})
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    time = np.arange(len(signal)) / sampling_rate
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, signal, 'k-', linewidth=0.5, label='ECG')
    
    if annotations is not None:
        colors = {
            'P': 'blue',
            'P_associated': 'blue',
            'P_dissociated': 'red',
            'QRS': 'green',
            'T': 'orange'
        }
        
        for wave_type, indices in annotations.items():
            color = colors.get(wave_type, 'gray')
            for idx in indices:
                ax.axvline(time[idx], color=color, alpha=0.5, linestyle='--', 
                          linewidth=1, label=wave_type if idx == indices[0] else "")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def calculate_pr_interval(
    p_peak_idx: int,
    qrs_onset_idx: int,
    sampling_rate: int = 500
) -> float:
    """
    Calculate PR interval in milliseconds
    
    Args:
        p_peak_idx: Index of P-wave peak
        qrs_onset_idx: Index of QRS onset
        sampling_rate: Sampling rate in Hz
        
    Returns:
        PR interval in milliseconds
    """
    samples_diff = qrs_onset_idx - p_peak_idx
    pr_interval_ms = (samples_diff / sampling_rate) * 1000
    return pr_interval_ms


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

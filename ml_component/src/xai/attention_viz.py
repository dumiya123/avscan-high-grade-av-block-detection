"""
Attention visualization module
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def extract_attention_weights(model, input_signal):
    """
    Extract attention weights from model
    
    Args:
        model: ECG U-Net model
        input_signal: Input ECG (1, 1, seq_len)
        
    Returns:
        Dictionary of attention maps
    """
    model.eval()
    
    with torch.no_grad():
        attention_maps = model.get_attention_maps(input_signal)
    
    return attention_maps


def plot_attention_timeline(ecg_signal, attention_map, save_path=None):
    """
    Plot attention weights over ECG timeline
    
    Args:
        ecg_signal: ECG signal (seq_len,)
        attention_map: Attention weights (seq_len,)
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    
    time = np.arange(len(ecg_signal)) / 500
    
    # Plot ECG
    ax1.plot(time, ecg_signal, 'k-', linewidth=0.8)
    ax1.set_ylabel('ECG Amplitude (mV)')
    ax1.set_title('ECG Signal')
    ax1.grid(True, alpha=0.3)
    
    # Plot attention
    ax2.plot(time, attention_map, 'r-', linewidth=1.5)
    ax2.fill_between(time, 0, attention_map, alpha=0.3, color='red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Attention Weight')
    ax2.set_title('Model Attention')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def identify_key_regions(attention_map, threshold=0.7):
    """
    Identify regions with high attention
    
    Args:
        attention_map: Attention weights (seq_len,)
        threshold: Threshold for high attention
        
    Returns:
        List of (start, end) tuples for high-attention regions
    """
    high_attention = attention_map > threshold
    
    # Find contiguous regions
    diff = np.diff(np.concatenate([[0], high_attention.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    regions = list(zip(starts, ends))
    
    return regions


if __name__ == "__main__":
    # Test attention visualization
    ecg_signal = np.random.randn(5000)
    attention_map = np.random.rand(5000)
    
    fig = plot_attention_timeline(ecg_signal, attention_map)
    print("✅ Attention visualization created")
    
    regions = identify_key_regions(attention_map, threshold=0.7)
    print(f"✅ Found {len(regions)} high-attention regions")

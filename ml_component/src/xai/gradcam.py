"""
Grad-CAM implementation for ECG U-Net
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM:
    """
    Grad-CAM for visualizing which parts of ECG influenced the model's decision
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute Grad-CAM on (e.g., last encoder block)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_signal, target_class=None, task='classification'):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_signal: Input ECG signal (1, 1, seq_len)
            target_class: Target class for gradient computation
            task: 'classification' or 'segmentation'
            
        Returns:
            Heatmap (seq_len,)
        """
        self.model.eval()
        
        # Forward pass
        seg_out, clf_out = self.model(input_signal)
        
        # Select output based on task
        if task == 'classification':
            if target_class is None:
                target_class = torch.argmax(clf_out, dim=1)
            score = clf_out[0, target_class]
        else:  # segmentation
            if target_class is None:
                target_class = 3  # QRS class
            score = seg_out[0, target_class, :].mean()
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (channels, seq_len)
        activations = self.activations[0]  # (channels, seq_len)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=1, keepdim=True)  # (channels, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)  # (seq_len,)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def overlay_on_ecg(self, ecg_signal, heatmap, alpha=0.4):
        """
        Create visualization of heatmap overlaid on ECG
        
        Args:
            ecg_signal: ECG signal (seq_len,)
            heatmap: Grad-CAM heatmap (seq_len,)
            alpha: Transparency of heatmap
            
        Returns:
            RGB image array
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Normalize ECG
        ecg_norm = (ecg_signal - ecg_signal.min()) / (ecg_signal.max() - ecg_signal.min())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 4))
        
        # Plot ECG
        time = np.arange(len(ecg_signal)) / 500  # Assuming 500 Hz
        ax.plot(time, ecg_signal, 'k-', linewidth=0.8, label='ECG')
        
        # Overlay heatmap
        heatmap_colored = cm.jet(heatmap)
        
        for i in range(len(time) - 1):
            ax.axvspan(time[i], time[i+1], alpha=alpha * heatmap[i], 
                      color=heatmap_colored[i, :3])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title('Grad-CAM Visualization')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


if __name__ == "__main__":
    from src.models.ecg_unet import ECGUNet
    
    # Test Grad-CAM
    model = ECGUNet()
    model.eval()
    
    # Get target layer (last encoder block)
    target_layer = model.enc4.conv
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    input_signal = torch.randn(1, 1, 5000)
    heatmap = gradcam.generate_heatmap(input_signal, task='classification')
    
    print(f"✅ Grad-CAM heatmap generated: shape {heatmap.shape}")
    print(f"   Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}")

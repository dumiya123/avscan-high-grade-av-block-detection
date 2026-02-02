"""
AtrionNet XAI: Grad-CAM (Gradient-weighted Class Activation Mapping)
This module provides the 'Visual Why' for the model's classification decisions.

Mathematics:
1.  **Forward Pass**: Compute the probability of the diagnosis (e.g., 3rd Degree Block).
2.  **Backprop**: Calculate gradients of the diagnosis with respect to the 
    last convolutional layer's features.
3.  **Importance Weighting**: Average the gradients across time to see which 
    feature channels were most important globally.
4.  **Heatmap Generation**: Multiply the feature maps by these weights and 
    sum them up. Only positive influences (ReLU) are kept to show what 
    *contributed* to the decision.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM:
    """
    Visual Explanation Engine.
    
    Why: Clinicians need to see which P-waves or QRS complexes the model 
    actually "looked at" when making a diagnosis. If the model is focusing 
    on baseline noise, the diagnosis is likely a false positive.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks: These allow us to 'catch' data flowing through the model 
        # without modifying the core model code.
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_signal, target_class=None, task='classification'):
        """
        The Heatmap Synthesis Pipeline.
        """
        self.model.eval()
        
        # 1. Prediction Stage
        seg_out, clf_out = self.model(input_signal)
        
        # 2. Select the 'Evidence' we want to explain
        if task == 'classification':
            if target_class is None:
                target_class = torch.argmax(clf_out, dim=1)
            score = clf_out[0, target_class]
        else:
            if target_class is None:
                target_class = 3
            score = seg_out[0, target_class, :].mean()
        
        # 3. Backpropagation Stage: Calculate how much each neuron in the 
        # target_layer contributed to the final score.
        self.model.zero_grad()
        score.backward()
        
        # 4. Feature Combination Stage
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # GAP (Global Average Pooling): Reduce gradients to a single weight per channel
        weights = torch.mean(gradients, dim=1, keepdim=True)
        
        # Linear Combination: Sum of [weights * activations]
        cam = torch.sum(weights * activations, dim=0)
        
        # ReLU: We only care about features that POSITIVELY impacted the score
        cam = F.relu(cam)
        
        # Normalization: Map to 0.0 - 1.0 for visualization
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

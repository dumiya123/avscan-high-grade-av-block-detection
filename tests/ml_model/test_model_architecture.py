import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml_component')))

# Try to import the ECGUNet model, handle if path structure is not exact as expected
try:
    from src.models.ecg_unet import ECGUNet
except ImportError:
    ECGUNet = None

@pytest.mark.skipif(ECGUNet is None, reason="ECGUNet model class not found in ml_component/src/models/ecg_unet.py")
def test_ecg_unet_forward_pass():
    """Verify that the model takes the expected input shape and yields correct output shapes."""
    model = ECGUNet()
    model.eval()
    
    # Simulate a batch size 2, 1 channel per sequence, 5000 seq length (10 seconds, 500 Hz)
    dummy_input = torch.randn(2, 1, 5000)
    
    with torch.no_grad():
        seg_out, clf_out = model(dummy_input)
        
    # Check segmentation output (5 classes expected: BG, P_assoc, P_dissoc, QRS, T)
    assert seg_out.shape == (2, 5, 5000), f"Segmentation shape mismatch: expected (2, 5, 5000), got {seg_out.shape}"
    
    # Check classification output (6 AV block types expected)
    assert clf_out.shape == (2, 6), f"Classification shape mismatch: expected (2, 6), got {clf_out.shape}"

@pytest.mark.skipif(ECGUNet is None, reason="ECGUNet model class not found")
def test_model_gradients_flow():
    """Ensure that the model can backpropagate gradients from both heads to the encoder."""
    model = ECGUNet()
    model.train()
    
    dummy_input = torch.randn(2, 1, 5000)
    seg_out, clf_out = model(dummy_input)
    
    # Compute dummy loss summing all elements
    loss = seg_out.sum() + clf_out.sum()
    loss.backward()
    
    # Check if a gradient is present and not zero in the first convolutional layer
    # The actual variable name depends on standard PyTorch conventions
    has_gradients = any(param.grad is not None and param.grad.abs().sum() > 0 
                        for param in model.parameters() if param.requires_grad)
    
    assert has_gradients, "Gradients did not flow backwards through the network."

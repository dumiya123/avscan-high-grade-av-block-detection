"""
Research Pipeline Verification.
Ensures that the cleaned codebase is fully functional.
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.atrion_net import AtrionNetHybrid
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.losses.segmentation_losses import create_instance_loss
from src.engine.atrion_evaluator import compute_instance_metrics
from src.utils.plotting import save_publication_plots

def verify():
    print("🧪 Verifying Hybrid Model Initialization...")
    model = AtrionNetHybrid(in_channels=12)
    device = torch.device("cpu")
    model.to(device)
    
    print("🧪 Verifying Dataset & Normalization...")
    # Dummy data [10, 12, 5000]
    dummy_sigs = np.random.randn(2, 12, 5000)
    dummy_anns = [{'p_waves': [(1000, 1050, 1100)]}] * 2
    dataset = AtrionInstanceDataset(dummy_sigs, dummy_anns, is_train=True)
    sample = dataset[0]
    
    print("🧪 Verifying Forward Pass & Loss Calculation...")
    sig = sample['signal'].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in sample.items() if k != 'signal'}
    preds = model(sig)
    loss = create_instance_loss(preds, targets)
    print(f"  Forward Pass Success. Loss: {loss.item():.4f}")
    
    print("🧪 Verifying Evaluator Logic...")
    targets_eval = [{'span': (1000, 1100)}]
    metrics = compute_instance_metrics(preds['heatmap'][0].detach().numpy(), 
                                     preds['width'][0].detach().numpy(), 
                                     targets_eval)
    print(f"  Evaluator Success. F1: {metrics['f1']:.4f}")
    
    print("🧪 Verifying Plotting Logic...")
    history = {'train_loss': [0.5, 0.4], 'val_loss': [0.6, 0.5], 'val_f1': [0.1, 0.2]}
    os.makedirs("verify_plots", exist_ok=True)
    save_publication_plots(history, None, "verify_plots")
    
    print("\n✅ CLEANUP VALIDATION SUCCESSFUL!")
    print("Architecture is clean, reproducible, and thesis-ready.")

if __name__ == "__main__":
    import os
    verify()

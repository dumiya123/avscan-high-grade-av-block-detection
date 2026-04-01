"""
Master Baseline Trainer — v3.0
==============================
Trains the two required academic benchmarks (Simple CNN & 1D U-Net) 
so they can be evaluated against AtrionNet in the final Ensemble Benchmarks.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Handle deeper directory structure for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import DATA_DIR, CHECKPOINT_DIR
from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.losses.segmentation_losses import create_instance_loss
from benchmarking.baselines.simple_cnn_baseline import SimpleCNNBaseline
from benchmarking.baselines.unet_1d_baseline import UNet1D_Baseline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 16

def train_model(model_class, model_name):
    print(f"\n🚀 TRAINING BASELINE: {model_name} ({DEVICE})")
    print("------------------------------------------")
    
    loader = LUDBLoader(DATA_DIR)
    signals, annotations = loader.get_all_data()
    total = len(signals)
    np.random.seed(42)
    indices = np.random.permutation(total)
    split = int(total * 0.85)
    
    train_ds = AtrionInstanceDataset([signals[i] for i in indices[:split]], 
                                    [annotations[i] for i in indices[:split]], is_train=True)
    val_ds = AtrionInstanceDataset([signals[i] for i in indices[split:]], 
                                  [annotations[i] for i in indices[split:]], is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = model_class(in_channels=12).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')
    save_path = os.path.join(CHECKPOINT_DIR, f"{model_name.lower()}_best.pth")

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            sig = batch['signal'].to(DEVICE)
            tgt = {k: v.to(DEVICE) for k, v in batch.items() if k != 'signal'}
            
            optimizer.zero_grad()
            out = model(sig)
            loss = create_instance_loss(out, tgt)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        # Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sig = batch['signal'].to(DEVICE)
                tgt = {k: v.to(DEVICE) for k, v in batch.items() if k != 'signal'}
                out = model(sig)
                v_loss += create_instance_loss(out, tgt).item()
        
        avg_v_loss = v_loss / len(val_loader)
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Saving new best for {model_name} (Loss: {avg_v_loss:.4f})")

    print(f"✅ {model_name} Training Complete. Weights saved to {save_path}")

if __name__ == '__main__':
    train_model(SimpleCNNBaseline, "Simple_CNN_Baseline")
    train_model(UNet1D_Baseline, "UNet1D_Baseline")

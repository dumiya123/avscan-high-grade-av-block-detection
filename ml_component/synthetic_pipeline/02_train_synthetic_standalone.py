"""
Isolated Synthetic Training Pipeline
====================================
This script trains AtrionNet EXCLUSIVELY on the synthetic 1,500-record dataset.
By design, this script does NOT import ludb_loader.py and does NOT touch the
original PhysioNet LUDB files. This ensures strict isolation according
to the thesis testing constraints (Regime B).

OUTPUT:
    - atrion_synthetic_best.pth
"""

import os
import sys
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.modeling.atrion_net import AtrionNetHybrid
from src.losses.segmentation_losses import create_instance_loss

# ── Configuration & Independence ──────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-4

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SIG_FILE = os.path.join(DATA_DIR, 'synthetic_signals.npy')
ANN_FILE = os.path.join(DATA_DIR, 'synthetic_annotations.pkl')
OUTPUT_MODEL = os.path.join(DATA_DIR, 'atrion_synthetic_best.pth')

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def load_isolated_dataset():
    """Loads only the synthetic data, ignoring LUDB entirely."""
    if not os.path.exists(SIG_FILE):
        raise FileNotFoundError("Synthetic signals not found. Run 01_synthetic_generator.py first.")
        
    signals = np.load(SIG_FILE)
    with open(ANN_FILE, 'rb') as f:
        annotations = pickle.load(f)
        
    print(f"📦 Loaded {len(signals)} isolated synthetic records.")
    
    # Random split: 80% Train, 20% Val (Testing is done later on LUDB)
    indices = np.random.permutation(len(signals))
    split = int(len(indices) * 0.8)
    idx_train, idx_val = indices[:split], indices[split:]
    
    train_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_train],
        [annotations[i] for i in idx_train],
        is_train=True
    )
    val_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_val],
        [annotations[i] for i in idx_val],
        is_train=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

def main():
    print("=" * 60)
    print("        REGIME B: SYNTHETIC-ONLY TRAINING")
    print("=" * 60)
    setup_seed(SEED)
    
    train_loader, val_loader = load_isolated_dataset()
    model = AtrionNetHybrid(in_channels=12).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    print("\n🚀 Beginning isolated training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            sig = batch['signal'].to(DEVICE)
            tgt = {k: v.to(DEVICE) for k, v in batch.items() if k != 'signal'}
            
            optimizer.zero_grad()
            out = model(sig)
            loss = create_instance_loss(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sig = batch['signal'].to(DEVICE)
                tgt = {k: v.to(DEVICE) for k, v in batch.items() if k != 'signal'}
                out = model(sig)
                loss = create_instance_loss(out, tgt)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_MODEL)
            
    print(f"\n✅ Training complete! Best isolated weights saved to: {OUTPUT_MODEL}")


if __name__ == '__main__':
    main()

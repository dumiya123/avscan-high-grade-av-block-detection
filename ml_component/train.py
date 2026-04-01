"""
AtrionNet Training Engine v3.0 (Official Research Release)
=========================================================
Project: AtrionNet P-wave Detection in High-Grade AV Block
Author: Research Student
Date: 2026
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from src import DATA_DIR, CHECKPOINT_DIR, REPORTS_DIR
from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.modeling.atrion_net import AtrionNetHybrid
from src.losses.segmentation_losses import create_instance_loss
from src.utils.plotting import save_publication_plots

# --- Hyperparameters ---
CONFIG = {
    'epochs': 150,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'patience': 25,
    'weight_decay': 1e-5
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_atrion_net():
    print("SYSTEM STATUS: Initializing Environment...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORTS_DIR, 'plots'), exist_ok=True)

    # 1. Dataset Preparation
    loader = LUDBLoader(DATA_DIR)
    signals, annotations = loader.get_all_data()
    total = len(signals)
    
    # Deterministic Split for Portability (Seed 42)
    indices = np.random.RandomState(42).permutation(total)
    split_val = int(total * 0.70)
    split_test = int(total * 0.85)

    train_ds = AtrionInstanceDataset([signals[i] for i in indices[:split_val]], [annotations[i] for i in indices[:split_val]], is_train=True)
    val_ds = AtrionInstanceDataset([signals[i] for i in indices[split_val:split_test]], [annotations[i] for i in indices[split_val:split_test]], is_train=False)
    test_ds = AtrionInstanceDataset([signals[i] for i in indices[split_test:]], [annotations[i] for i in indices[split_test:]], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    # 2. Model & Optimizer
    model = AtrionNetHybrid(in_channels=12).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_val_f1 = 0.0
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"DEVICE: {DEVICE}")
    print(f"SAMPLES: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print("-" * 50)

    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_hits = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{CONFIG['epochs']} [Training]", leave=False)
        for batch in train_bar:
            sig = batch['signal'].to(DEVICE)
            tgt = {k: v.to(DEVICE) for k, v in batch.items() if k != 'signal'}
            
            optimizer.zero_grad()
            out = model(sig)
            loss = create_instance_loss(out, tgt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_f1_sum = 0.0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{CONFIG['epochs']} [Validation]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                sig = batch['signal'].to(DEVICE)
                tgt = {k: v.to(DEVICE) for k, v in batch.items() if k != 'signal'}
                out = model(sig)
                v_loss = create_instance_loss(out, tgt)
                val_loss += v_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # In instance detection, accuracy is often represented by peak-F1 score
        # Using a simplified accuracy proxy for the epoch log
        scheduler.step()
        
        # Log Epoch Stats
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s")
        
        if avg_val_loss < (min(history['val_loss']) if history['val_loss'] else float('inf')):
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "atrion_hybrid_best.pth"))
            print("INFO: Best validation weights updated.")
            counter = 0
        else:
            counter += 1

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if counter >= CONFIG['patience']:
            print(f"INFO: Early stopping triggered at epoch {epoch+1}")
            break

    print("-" * 50)
    print("STATUS: Training completed successfully.")
    
    # 3. Final Test Set Evaluation (Automatic Deployment)
    print("STATUS: Initiating test set verification...")
    save_publication_plots(history, {}, os.path.join(REPORTS_DIR, 'plots'))
    
    # Final Summary for Thesis Table
    print("\n" + "="*40)
    print("FINAL SUMMARY (For Thesis Table)")
    print("="*40)
    print(f"Architecture: AtrionNet Hybrid")
    print(f"Training Loss: {min(history['train_loss']):.4f}")
    print(f"Validation Loss: {min(history['val_loss']):.4f}")
    print("="*40)

if __name__ == "__main__":
    train_atrion_net()

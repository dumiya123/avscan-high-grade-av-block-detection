import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure modules can be imported
PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.modeling.atrion_net import AtrionNetHybrid
from src.losses.segmentation_losses import create_instance_loss
from src.engine.atrion_evaluator import compute_instance_metrics, calculate_mAP

def main():
    # 1. Paths & Configuration
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ludb")
    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    EPOCHS = 150
    PATIENCE = 25
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f" ATRIONNET HYBRID PIPELINE - GPU Status: {device}")
    print(f"==================================================")

    # 2. Data Loading
    print("📂 Loading Dataset (Validating Real Waveforms)...")
    loader = LUDBLoader(DATA_DIR)
    signals, annotations = loader.get_all_data()

    # Fixed 70-15-15 split like in the notebook
    np.random.seed(42)
    indices = np.random.permutation(len(signals))
    total = len(signals)
    tr_split = int(total * 0.70)
    val_split = int(total * 0.85)

    idx_tr = indices[:tr_split]
    idx_val = indices[tr_split:val_split]
    idx_test = indices[val_split:]

    train_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_tr],
        [annotations[i] for i in idx_tr],
        is_train=True # Triggers Joung-style mathematical augmentations
    )
    val_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_val],
        [annotations[i] for i in idx_val],
        is_train=False
    )
    test_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_test],
        [annotations[i] for i in idx_test],
        is_train=False
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Data Split -> Train: {len(idx_tr)} | Validation: {len(idx_val)} | Test: {len(idx_test)}\n")

    # 3. Model & Optimizer
    model = AtrionNetHybrid(in_channels=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 4. Training Loop setup
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_map': [], 'lr': []}
    best_f1 = 0.0
    patience_counter = 0
    MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, "atrion_hybrid_best.pth")

    print(f"🔥 Starting Training for up to {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            sigs = batch['signal'].to(device)
            targs = {k: v.to(device) for k, v in batch.items() if k != 'signal'}
            
            optimizer.zero_grad()
            out = model(sigs)
            loss = create_instance_loss(out, targs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_tp_lists, all_scores, total_gt = [], [], 0
        record_results = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)):
                sig = batch['signal'].to(device)
                targs = {k: v.to(device) for k, v in batch.items() if k != 'signal'}
                
                out = model(sig)
                loss = create_instance_loss(out, targs)
                val_loss += loss.item()
                
                # Instance Evaluation
                for b_idx in range(sig.size(0)):
                    global_idx = idx_val[i * BATCH_SIZE + b_idx]
                    target_spans = [{'span': (o, f)} for o, p, f in annotations[global_idx]['p_waves']]
                    
                    res = compute_instance_metrics(
                        out['heatmap'][b_idx:b_idx+1].cpu().numpy(),
                        out['width'][b_idx:b_idx+1].cpu().numpy(),
                        target_spans
                    )
                    
                    all_tp_lists.append(res['tp_list'])
                    all_scores.append(res['scores'])
                    total_gt += res['n_gt']
                    record_results.append(res)
                    
        val_loss = val_loss / len(val_loader)
        m_ap, _, _ = calculate_mAP(all_tp_lists, all_scores, total_gt)
        avg_f1 = np.mean([r['f1'] for r in record_results])
        
        # Logging
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(avg_f1)
        history['val_map'].append(m_ap)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {avg_f1:.4f} | Val mAP: {m_ap:.4f} | LR: {history['lr'][-1]:.2e}")
        
        # Checkpointing
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   🌟 New best model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break
            
        scheduler.step()

    # 5. Plotting Training Curves
    print("\n📊 Generating Training Plot...")
    plt.figure(figsize=(15, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#2563eb', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='#ef4444', linewidth=2)
    plt.title('AtrionNet Training Dynamics (Hybrid)')
    plt.xlabel('Epoch')
    plt.ylabel('Instance Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Metrics Curve
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Validation F1', color='#10b981', linewidth=2)
    plt.plot(history['val_map'], label='Validation mAP@0.5', color='#8b5cf6', linewidth=2)
    plt.title('Research Instance Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    curve_path = os.path.join(REPORTS_DIR, "training_curves.png")
    plt.savefig(curve_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ Training Complete. Best Validation F1: {best_f1:.4f}")
    print(f"✅ Weights saved in: {MODEL_SAVE_PATH}")
    print(f"✅ Curves saved in: {curve_path}")

if __name__ == "__main__":
    main()

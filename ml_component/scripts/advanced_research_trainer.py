"""
ADVANCED RESEARCH TRAINER - AtrionNet v3.5 (Hybrid Edition)
Specifically designed for High-Grade AV Block Thesis results.
"""
import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.atrion_net import AtrionNetHybrid
from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.losses.segmentation_losses import create_instance_loss
from src.engine.atrion_evaluator import compute_instance_metrics, calculate_mAP
from src.utils.plotting import save_publication_plots, plot_confusion_matrix, plot_pr_curve

# Configuration
CONFIG = {
    'data_dir': "data/raw/ludb",
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'epochs': 150,
    'batch_size': 16,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,
    'seed': 42,
    'plot_dir': "reports/plots",
    'model_save_path': "weights/atrion_hybrid_best.pth",
    'test_indices_path': "data/processed/test_indices.npy"
}

os.makedirs(CONFIG['plot_dir'], exist_ok=True)
os.makedirs("weights", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    set_seed(CONFIG['seed'])
    
    # 1. Load Data
    loader = LUDBLoader(CONFIG['data_dir'])
    signals, annotations = loader.get_all_data()
    
    indices = np.arange(len(signals))
    np.random.shuffle(indices)
    tr_stop = int(0.8 * len(indices))
    v_stop = int(0.9 * len(indices))
    
    idx_tr, idx_val, idx_test = indices[:tr_stop], indices[tr_stop:v_stop], indices[v_stop:]
    np.save(CONFIG['test_indices_path'], idx_test)
    
    train_ds = AtrionInstanceDataset(signals[idx_tr], [annotations[i] for i in idx_tr], is_train=True)
    val_ds = AtrionInstanceDataset(signals[idx_val], [annotations[i] for i in idx_val], is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    
    # 2. Hybrid Model Initialization
    model = AtrionNetHybrid(in_channels=12).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    history = {
        'train_loss': [], 'val_loss': [], 'lr': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_map': []
    }
    
    best_f1 = 0
    patience_counter = 0
    
    print(f"\n🚀 STARTING RESEARCH TRAINING | HYBRID ARCHITECTURE")
    print(f"📊 Training Pool: {len(idx_tr)} records | Device: {CONFIG['device']}")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        t_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            sigs = batch['signal'].to(CONFIG['device'])
            targs = {k: v.to(CONFIG['device']) for k, v in batch.items() if k != 'signal'}
            
            optimizer.zero_grad()
            preds = model(sigs)
            loss = create_instance_loss(preds, targs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            
        # Validation
        model.eval()
        v_loss = 0
        all_tp_lists, all_scores, total_gt = [], [], 0
        v_prec, v_rec, v_f1 = [], [], []
        
        with torch.no_grad():
            val_idx_counter = 0
            for batch in val_loader:
                sigs = batch['signal'].to(CONFIG['device'])
                targs = {k: v.to(CONFIG['device']) for k, v in batch.items() if k != 'signal'}
                out = model(sigs)
                v_loss += create_instance_loss(out, targs).item()
                
                # Instance Metrics for each record in batch
                for i in range(len(sigs)):
                    actual_idx = idx_val[val_idx_counter]
                    targets = [{'span': (o, f)} for o, p, f in annotations[actual_idx]['p_waves']]
                    
                    res = compute_instance_metrics(out['heatmap'][i:i+1].cpu().numpy(), 
                                                out['width'][i:i+1].cpu().numpy(), 
                                                targets)
                    
                    all_tp_lists.append(res['tp_list'])
                    all_scores.append(res['scores'])
                    total_gt += res['n_gt']
                    v_prec.append(res['precision'])
                    v_rec.append(res['recall'])
                    v_f1.append(res['f1'])
                    val_idx_counter += 1
        
        m_ap, val_rec_curve, val_pre_curve = calculate_mAP(all_tp_lists, all_scores, total_gt)
        avg_f1 = np.mean(v_f1)
        
        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(v_loss/len(val_loader))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['val_precision'].append(np.mean(v_prec))
        history['val_recall'].append(np.mean(v_rec))
        history['val_f1'].append(avg_f1)
        history['val_map'].append(m_ap)
        
        print(f"✅ [Epoch {epoch+1:03d}] Loss: {history['val_loss'][-1]:.4f} | F1: {avg_f1:.4f} | mAP: {m_ap:.4f}")
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            patience_counter = 0
            # Save temporary PR curve data for best model
            best_pr = (val_rec_curve, val_pre_curve, m_ap)
            print("⭐ New Best Model Saved!")
        else:
            patience_counter += 1
            
        scheduler.step()
        if patience_counter >= CONFIG['early_stopping_patience']:
            print("🛑 Early Stopping Triggered.")
            break
            
    # Final Visualization
    print("\n🔬 Generating Publication Evidence...")
    save_publication_plots(history, None, CONFIG['plot_dir'])
    if 'best_pr' in locals():
        plot_pr_curve(best_pr[0], best_pr[1], best_pr[2], f"{CONFIG['plot_dir']}/pr_curve_val_best.png")
    
    print(f"📊 Results saved to {CONFIG['plot_dir']}")

if __name__ == "__main__":
    train()

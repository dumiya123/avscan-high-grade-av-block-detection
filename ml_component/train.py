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

# CENTRALIZED HYPERPARAMETERS
# Restored to Balanced Physiological Defaults to maximize F1-Score and prevent False Positive collapse
ATRION_CONFIG = {
    # Training Loop Context
    "EPOCHS": 150,
    "PATIENCE": 25,
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-4,
    
    # Instance Mathematical Balancing (Focal Loss)
    "FOCAL_ALPHA": 2.0,     # Standard optimized Focal Loss alpha
    "FOCAL_BETA": 4.0,
    "LOSS_WEIGHT_HM": 2.0,  # Balanced detection and segmentation weighting
    "LOSS_WEIGHT_W": 1.0,
    "LOSS_WEIGHT_M": 1.0,
    
    # Universal Inference Bounds (Valid Physiological Floor)
    "EVAL_CONF": 0.45,      # Raised to match tighter sigma=6 heatmap targets
    "EVAL_DIST": 60,        # Minimum 120ms inter-peak distance
    "EVAL_PROM": 0.10       # 10% structural prominence required
}

def main(drive_backup=None):
    # 1. Paths & Configuration
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ludb")
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
    WEIGHTS_DIR = os.path.join(OUTPUTS_DIR, "weights")
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
    CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    EPOCHS = ATRION_CONFIG['EPOCHS']
    PATIENCE = ATRION_CONFIG['PATIENCE']
    BATCH_SIZE = ATRION_CONFIG['BATCH_SIZE']
    LEARNING_RATE = ATRION_CONFIG['LEARNING_RATE']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[AtrionNet] Training Pipeline Initialized (Device: {device})")

    # 2. Data Loading
    print("[AtrionNet] Loading LUDB Dataset...")
    loader = LUDBLoader(DATA_DIR)
    signals, annotations = loader.get_all_data()

    # Fixed 70-15-15 split
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Data Split -> Train: {len(idx_tr)} | Validation: {len(idx_val)} | Test: {len(idx_test)}\n")

    # 3. Model & Optimizer
    model = AtrionNetHybrid(in_channels=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Using Plateau scheduler to avoid stopping at minimum LR of Cosine cycle
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    # 4. Training Loop setup
    # Track separate precision and recall
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_map': [], 'lr': []}
    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 0
    MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, "atrion_hybrid_best.pth")
    LATEST_CKPT_PATH = os.path.join(CHECKPOINTS_DIR, "latest_checkpoint.pth")
    ACCUMULATION_STEPS = 4

    # === Checkpoint Resuming Logic ===
    import shutil
    if drive_backup is not None:
        drive_ckpt_path = os.path.join(drive_backup, "latest_checkpoint.pth")
        if os.path.exists(drive_ckpt_path):
            print(f"\n🔄 Found Cloud Backup at {drive_backup}! Downloading to local workspace...")
            shutil.copy2(drive_ckpt_path, LATEST_CKPT_PATH)

    if os.path.exists(LATEST_CKPT_PATH):
        print(f"\n📂 Checkpoint found at {LATEST_CKPT_PATH}! Restoring previous training state...")
        try:
            checkpoint_data = torch.load(LATEST_CKPT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            start_epoch = checkpoint_data['epoch'] + 1
            best_f1 = checkpoint_data['best_f1']
            history = checkpoint_data['history']
            patience_counter = checkpoint_data['patience_counter']
            print(f"✅ Successfully resumed! Continuing from Epoch {start_epoch+1}.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint. Starting from scratch. Error: {e}")
    # ==================================

    print(f"\n[AtrionNet] Starting Training Loop (Max Epochs: {EPOCHS})...\n")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        # Training with Gradient Accumulation
        for b_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)):
            sigs = batch['signal'].to(device)
            targs = {k: v.to(device) for k, v in batch.items() if k != 'signal'}
            
            out = model(sigs)
            loss = create_instance_loss(
                out, targs,
                alpha=ATRION_CONFIG['FOCAL_ALPHA'],
                beta=ATRION_CONFIG['FOCAL_BETA'],
                hm_weight=ATRION_CONFIG['LOSS_WEIGHT_HM'],
                w_weight=ATRION_CONFIG['LOSS_WEIGHT_W'],
                m_weight=ATRION_CONFIG['LOSS_WEIGHT_M']
            )
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if ((b_idx + 1) % ACCUMULATION_STEPS == 0) or (b_idx + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * ACCUMULATION_STEPS
            
        train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_tp_lists, all_scores, total_gt = [], [], 0
        total_tp, total_fp, total_fn = 0, 0, 0
        
        val_sample_idx = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)):
                sig = batch['signal'].to(device)
                targs = {k: v.to(device) for k, v in batch.items() if k != 'signal'}
                
                out = model(sig)
                loss = create_instance_loss(
                    out, targs,
                    alpha=ATRION_CONFIG['FOCAL_ALPHA'],
                    beta=ATRION_CONFIG['FOCAL_BETA'],
                    hm_weight=ATRION_CONFIG['LOSS_WEIGHT_HM'],
                    w_weight=ATRION_CONFIG['LOSS_WEIGHT_W'],
                    m_weight=ATRION_CONFIG['LOSS_WEIGHT_M']
                )
                val_loss += loss.item()
                
                # Instance Evaluation
                for b_j in range(sig.size(0)):
                    global_idx = idx_val[val_sample_idx]
                    target_spans = [{'span': (o, f)} for o, p, f in annotations[global_idx]['p_waves']]
                    
                    res = compute_instance_metrics(
                        out['heatmap'][b_j:b_j+1].cpu().numpy(),
                        out['width'][b_j:b_j+1].cpu().numpy(),
                        target_spans,
                        conf_threshold=ATRION_CONFIG['EVAL_CONF'],
                        distance=ATRION_CONFIG['EVAL_DIST'],
                        prominence=ATRION_CONFIG['EVAL_PROM']
                    )
                    
                    all_tp_lists.append(res['tp_list'])
                    all_scores.append(res['scores'])
                    total_gt += res['n_gt']
                    total_tp += res['tp']
                    total_fp += res['fp']
                    total_fn += res['fn']
                    val_sample_idx += 1
                    
        val_loss = val_loss / len(val_loader)
        m_ap, _, _ = calculate_mAP(all_tp_lists, all_scores, total_gt)
        
        # Calculate Micro-Averaged global metrics
        global_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        global_f1 = 2 * global_prec * global_rec / (global_prec + global_rec) if (global_prec + global_rec) > 0 else 0.0
        
        # Logging
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_precision'].append(global_prec)
        history['val_recall'].append(global_rec)
        history['val_f1'].append(global_f1)
        history['val_map'].append(m_ap)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {global_f1:.4f} | Val mAP: {m_ap:.4f} | LR: {history['lr'][-1]:.2e}")
        
        # Scheduler Step
        scheduler.step(global_f1)
        
        # Checkpointing (Now micro-averaged F1)
        if global_f1 > best_f1:
            best_f1 = global_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   🌟 New best model saved!")
        else:
            patience_counter += 1
            
        # === Checkpoint Saving Mechanism ===
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'history': history,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint_data, LATEST_CKPT_PATH)
        
        # Sync immediately to Drive if provided
        if drive_backup is not None:
            os.makedirs(drive_backup, exist_ok=True)
            drive_ckpt_path = os.path.join(drive_backup, "latest_checkpoint.pth")
            shutil.copy2(LATEST_CKPT_PATH, drive_ckpt_path)
            # Also safely backup best weights to cloud continuously
            if global_f1 == best_f1:
                shutil.copy2(MODEL_SAVE_PATH, os.path.join(drive_backup, "atrion_hybrid_best.pth"))
        # =================================

        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Plotting Training Curves
    print("\n[AtrionNet] Generating Training Plots...")
    from src.utils.plotting import save_publication_plots, plot_confusion_matrix, plot_pr_curve
    save_publication_plots(history, None, PLOTS_DIR)
    
    # 6. Test Set Evaluation
    print("\n[AtrionNet] Running Final Evaluation on Test Dataset...")
    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    
    all_tp_lists_test, all_scores_test, total_gt_test = [], [], 0
    total_tp_test, total_fp_test, total_fn_test = 0, 0, 0
    
    test_sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            sig = batch['signal'].to(device)
            targs = {k: v.to(device) for k, v in batch.items() if k != 'signal'}
            
            out = model(sig)
            
            for b_j in range(sig.size(0)):
                global_idx = idx_test[test_sample_idx]
                target_spans = [{'span': (o, f)} for o, p, f in annotations[global_idx]['p_waves']]
                
                res = compute_instance_metrics(
                    out['heatmap'][b_j:b_j+1].cpu().numpy(),
                    out['width'][b_j:b_j+1].cpu().numpy(),
                    target_spans,
                    conf_threshold=ATRION_CONFIG['EVAL_CONF'],
                    distance=ATRION_CONFIG['EVAL_DIST'],
                    prominence=ATRION_CONFIG['EVAL_PROM']
                )
                
                all_tp_lists_test.append(res['tp_list'])
                all_scores_test.append(res['scores'])
                total_gt_test += res['n_gt']
                total_tp_test += res['tp']
                total_fp_test += res['fp']
                total_fn_test += res['fn']
                test_sample_idx += 1

    test_map, recalls, precisions = calculate_mAP(all_tp_lists_test, all_scores_test, total_gt_test)
    test_prec = total_tp_test / (total_tp_test + total_fp_test) if (total_tp_test + total_fp_test) > 0 else 0.0
    test_rec = total_tp_test / (total_tp_test + total_fn_test) if (total_tp_test + total_fn_test) > 0 else 0.0
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0.0

    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION REPORT")
    print("="*50)
    print(f"Total Targets (Ground Truth) : {total_gt_test}")
    print(f"True Positives (TP)          : {total_tp_test}")
    print(f"False Positives (FP)         : {total_fp_test}")
    print(f"False Negatives (FN)         : {total_fn_test}")
    print("-" * 50)
    print(f"Precision                    : {test_prec:.4f}")
    print(f"Recall                       : {test_rec:.4f}")
    print(f"F1-Score                     : {test_f1:.4f}")
    print(f"mAP @ 0.5                    : {test_map:.4f}")
    print("="*50)
    
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    pr_path = os.path.join(PLOTS_DIR, "pr_curve.png")
    
    plot_confusion_matrix(total_tp_test, total_fp_test, total_fn_test, cm_path)
    plot_pr_curve(recalls, precisions, test_map, pr_path)

    print(f"\n[AtrionNet] Training & Evaluation Complete.")
    print(f"--> Best Weights: {MODEL_SAVE_PATH}")
    print(f"--> Test Plots  : {PLOTS_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive_backup', type=str, default=None, help='Path to sync checkpoints natively')
    args = parser.parse_args()
    main(drive_backup=args.drive_backup)

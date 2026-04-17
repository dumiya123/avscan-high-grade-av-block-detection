import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
import json

# Ensure modules can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_COMP_DIR = os.path.join(PROJECT_ROOT, "ml_component")
if ML_COMP_DIR not in sys.path:
    sys.path.insert(0, ML_COMP_DIR)

from ml_component.src.data_pipeline.ludb_loader import LUDBLoader
from ml_component.src.data_pipeline.instance_dataset import AtrionInstanceDataset
from ml_component.src.losses.segmentation_losses import create_instance_loss
from ml_component.src.engine.atrion_evaluator import compute_instance_metrics, calculate_mAP
from ml_component.src.modeling.atrion_net import AtrionNetBaseline # U-Net
from ml_component.benchmarking_codes.baseline_models import CNN1DBaseline, CNNLSTMBaseline

# Shared configuration
CONFIG = {
    "EPOCHS": 50, # Fewer epochs for baselines or set to regular if time permits
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-4,
    "FOCAL_ALPHA": 2.0,
    "FOCAL_BETA": 4.0,
    "LOSS_WEIGHT_HM": 2.0,
    "LOSS_WEIGHT_W": 1.0,
    "LOSS_WEIGHT_M": 1.0,
    "EVAL_CONF": 0.45,
    "EVAL_DIST": 60,
    "EVAL_PROM": 0.10
}

def train_baseline(model_name, model, train_loader, val_loader, test_loader, device, annotations, idx_val, idx_test):
    print(f"\n{'='*50}\nStarting Training for {model_name}\n{'='*50}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    
    best_f1 = 0
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "ml_component", "benchmarking_codes", "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    best_model_path = os.path.join(OUTPUTS_DIR, f"{model_name}_best.pth")
    # For simplicity, we just run a few epochs or a full loop
    # This is a simplified training loop focused on producing benchmark models
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train {model_name} Epoch {epoch+1}", leave=False):
            sigs = batch['signal'].to(device)
            targs = {k: v.to(device) for k, v in batch.items() if k != 'signal'}
            
            optimizer.zero_grad()
            out = model(sigs)
            loss = create_instance_loss(out, targs, **{k.lower()[12:]: v for k, v in CONFIG.items() if k.startswith("LOSS_WEIGHT_") or k.startswith("FOCAL_")})
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        total_tp, total_fp, total_fn = 0, 0, 0
        val_sample_idx = 0
        with torch.no_grad():
            for batch in val_loader:
                sig = batch['signal'].to(device)
                out = model(sig)
                
                for b_j in range(sig.size(0)):
                    global_idx = idx_val[val_sample_idx]
                    target_spans = [{'span': (o, f)} for o, p, f in annotations[global_idx]['p_waves']]
                    res = compute_instance_metrics(out['heatmap'][b_j:b_j+1].cpu().numpy(), out['width'][b_j:b_j+1].cpu().numpy(),
                        target_spans, conf_threshold=CONFIG['EVAL_CONF'], distance=CONFIG['EVAL_DIST'], prominence=CONFIG['EVAL_PROM'])
                    total_tp += res['tp']
                    total_fp += res['fp']
                    total_fn += res['fn']
                    val_sample_idx += 1
                    
        val_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        val_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0.0
        
        print(f"Epoch {epoch+1} | {model_name} Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            
    print(f"Finished training {model_name}. Best F1: {best_f1}")
    
    # Returning the path to the best model
    return best_model_path

def run_benchmarks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    DATA_DIR = os.path.join(PROJECT_ROOT, "ml_component", "data", "raw", "ludb")
    loader = LUDBLoader(DATA_DIR)
    
    try:
        signals, annotations = loader.get_all_data()
    except Exception as e:
        print(f"Error loading data: {e}. Benchmarking script needs valid data at {DATA_DIR}")
        return

    import numpy as np
    np.random.seed(42)
    indices = np.random.permutation(len(signals))
    total = len(signals)
    tr_split = int(total * 0.70)
    val_split = int(total * 0.85)

    idx_tr = indices[:tr_split]
    idx_val = indices[tr_split:val_split]
    idx_test = indices[val_split:]

    train_ds = AtrionInstanceDataset([signals[i] for i in idx_tr], [annotations[i] for i in idx_tr], is_train=True)
    val_ds = AtrionInstanceDataset([signals[i] for i in idx_val], [annotations[i] for i in idx_val], is_train=False)
    test_ds = AtrionInstanceDataset([signals[i] for i in idx_test], [annotations[i] for i in idx_test], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    
    models_to_test = {
        "UNet_Baseline": AtrionNetBaseline(in_channels=12).to(device),
        "CNN_Baseline": CNN1DBaseline(in_channels=12).to(device),
        "CNN_LSTM_Baseline": CNNLSTMBaseline(in_channels=12).to(device)
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        best_path = train_baseline(name, model, train_loader, val_loader, test_loader, device, annotations, idx_val, idx_test)
        results[name] = best_path

    # Save tracking json
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "ml_component", "benchmarking_codes", "outputs")
    with open(os.path.join(OUTPUTS_DIR, "benchmark_models.json"), "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_benchmarks()

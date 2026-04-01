"""
Evaluation Protocol: Regime B vs LUDB Test Set
==============================================
Evaluates the model trained exclusively on synthetic data against
the strict 15% held-out test split of the original, real LUDB dataset.
This proves generalization from synthetic dissociation directly to human pathology.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import DATA_DIR, CHECKPOINT_DIR
from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.engine.atrion_evaluator import compute_instance_metrics, calculate_mAP, get_instances_from_heatmap

# ── Configuration & Independence ──────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'atrion_synthetic_best.pth')

def main():
    print("=" * 60)
    print("      REGIME B BENCHMARK: SYNTHETIC MODEL vs LUDB TEST")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Synthetic model weights not found. Run 02_train_synthetic_standalone.py first.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Load the exact same LUDB Test Split as Regime A
    print("📂 Loading exact 15% LUDB Test Split (SEED 42)...")
    loader = LUDBLoader(DATA_DIR)
    signals, annotations = loader.get_all_data()
    
    if len(signals) == 0:
        print(f"\n❌ ERROR: No LUDB data found in {DATA_DIR}")
        print("Please ensure you have run 'download_data.py' or check your Drive path.")
        sys.exit(1)
        
    total = len(signals)
    indices = np.random.permutation(total)
    idx_test = indices[int(total * 0.85):]
    
    test_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_test],
        [annotations[i] for i in idx_test],
        is_train=False
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 2. Load the isolated synthetic weights
    from src.modeling.atrion_net import AtrionNetHybrid
    model = AtrionNetHybrid(in_channels=12).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Evaluate
    print("⚖️  Running instance-level evaluation...")
    all_prec, all_rec, all_f1 = [], [], []
    all_scores, all_labels = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sig = batch['signal'].to(DEVICE)
            out = model(sig)
            
            global_idx = idx_test[i]
            target_pwaves = annotations[global_idx]['p_waves']
            target_spans = [{'span': (o, f), 'confidence': 1.0} for (o, p, f) in target_pwaves]
            
            hm = out['heatmap'][0, 0].cpu().numpy()
            wm = out['width'][0, 0].cpu().numpy()
            
            # Use proper metric computation API from atrion_evaluator.py
            res = compute_instance_metrics(hm, wm, target_spans, iou_threshold=0.5, conf_threshold=0.35)
            
            all_prec.append(res['precision'])
            all_rec.append(res['recall'])
            all_f1.append(res['f1'])
            
            # TP list holds 1 for TP and 0 for FP for each predicted score
            all_scores.extend(res['scores'])
            all_labels.extend(res['tp_list'])
                
    # calculate_mAP expects list of numpy arrays
    ap, _, _ = calculate_mAP([np.array(all_labels)], [np.array(all_scores)], sum(len(a['p_waves']) for a in [annotations[i] for i in idx_test]))
    
    print("\n" + "=" * 60)
    print("          FINAL BENCHMARK SCORE (REGIME B)")
    print("=" * 60)
    print(f"   Precision : {np.mean(all_prec):.4f}")
    print(f"   Recall    : {np.mean(all_rec):.4f}")
    print(f"   F1 Score  : {np.mean(all_f1):.4f}   <-- Compare this to Regime A (65%)")
    print(f"   mAP@0.5   : {ap:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main()

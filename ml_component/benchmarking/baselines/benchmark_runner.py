"""
AtrionNet Advanced Benchmarking — v6.0 (Ensemble Support)
========================================================
Includes:
  - AtrionNet Hybrid (Main Model)
  - Simple CNN (Baseline)
  - 1D U-Net (Segmentation Baseline)
  - Ensemble (Majority Voting)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Handle deeper directory structure for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import DATA_DIR, CHECKPOINT_DIR
from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.modeling.atrion_net import AtrionNetHybrid
from benchmarking.baselines.simple_cnn_baseline import SimpleCNNBaseline
from benchmarking.baselines.unet_1d_baseline import UNet1D_Baseline
from src.engine.atrion_evaluator import compute_instance_metrics, calculate_mAP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(current_dir, '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmarks():
    print("\n🔥 BOOTING ADVANCED ENSEMBLE BENCHMARK REGIME 6.0")
    print("--------------------------------------------------\n")

    # 1. Loading exact 15% TEST Split (reproducible with SEED 42)
    loader = LUDBLoader(DATA_DIR)
    signals, annotations_raw = loader.get_all_data()
    total = len(signals)
    np.random.seed(42)
    indices = np.random.permutation(total)
    idx_test = indices[int(total * 0.85):]
    
    test_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_test],
        [annotations_raw[i] for i in idx_test],
        is_train=False
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 2. Loading All Model Weights
    models = {
        'AtrionNet_Hybrid': AtrionNetHybrid(in_channels=12),
        'Simple_CNN_Baseline': SimpleCNNBaseline(in_channels=12),
        'UNet1D_Baseline': UNet1D_Baseline(in_channels=12)
    }
    
    for name, model in models.items():
        model.to(DEVICE)
        weight_path = os.path.join(CHECKPOINT_DIR, f"{name.lower() if 'Atrion' not in name else 'atrion_hybrid'}_best.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        else:
            print(f"⚠️ Warning: Weights not found for {name} at {weight_path}")
        model.eval()

    # 3. EVALUATION LOOP
    results = []
    
    with torch.no_grad():
        for name, model in models.items():
            print(f"🔬 Evaluating {name}...")
            all_tp_lists, all_scores, total_gt = [], [], 0
            record_metrics = []

            for i, batch in enumerate(test_loader):
                sig = batch['signal'].to(DEVICE)
                out = model(sig)
                
                # Fetch ground truth for this sample
                global_idx = idx_test[i]
                target_spans = [{'span': (o, f)} for o, p, f in annotations_raw[global_idx]['p_waves']]
                
                # Compute metrics
                res = compute_instance_metrics(
                    out['heatmap'][0:1].cpu().numpy(),
                    out['width'][0:1].cpu().numpy(),
                    target_spans
                )
                
                all_tp_lists.append(res['tp_list'])
                all_scores.append(res['scores'])
                total_gt += res['n_gt']
                record_metrics.append(res)
            
            m_ap, _, _ = calculate_mAP(all_tp_lists, all_scores, total_gt)
            f1_avg = np.mean([r['f1'] for r in record_metrics])
            prec_avg = np.mean([r['precision'] for r in record_metrics])
            rec_avg = np.mean([r['recall'] for r in record_metrics])

            results.append({
                'model': name,
                'precision': prec_avg,
                'recall': rec_avg,
                'f1': f1_avg,
                'mAP': m_ap
            })

    # 4. SAVE CSV
    df = pd.DataFrame(results)
    save_path = os.path.join(RESULTS_DIR, 'all_results.csv')
    df.to_csv(save_path, index=False)
    print(f"\n✅ ALL BENCHMARKS SAVED TO: {save_path}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    run_benchmarks()

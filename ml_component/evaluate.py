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
from src.engine.atrion_evaluator import compute_instance_metrics, calculate_mAP
from src.utils.plotting import plot_confusion_matrix

def main():
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ludb")
    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, "atrion_hybrid_best.pth")
    if not os.path.exists(MODEL_SAVE_PATH):
        print("❌ Error: No trained model weights found. Run train.py first!")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f" ATRIONNET HYBRID PIPELINE - GPU Status: {device}")
    print(f"==================================================")

    # 1. Data Loading (Test Split Only)
    print("📂 Loading Dataset...")
    loader = LUDBLoader(DATA_DIR)
    signals, annotations = loader.get_all_data()

    # Identical 70-15-15 split to ensure test set purity
    np.random.seed(42)
    indices = np.random.permutation(len(signals))
    test_split = int(len(signals) * 0.85)
    idx_test = indices[test_split:]

    test_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_test],
        [annotations[i] for i in idx_test],
        is_train=False
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test Split Size: {len(idx_test)}\n")

    # 2. Model Loading
    print("🧠 Loading Trained Weights...")
    model = AtrionNetHybrid(in_channels=12).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 3. Evaluation
    print("🔍 Running Final Evaluation on Held-out Test Set...")
    all_tp_lists, all_scores, total_gt = [], [], 0
    total_tp, total_fp, total_fn = 0, 0, 0
    record_results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            sig = batch['signal'].to(device)
            out = model(sig)
            
            # Reconstruct global index to fetch ground-truth spans
            global_idx = idx_test[i]
            target_spans = [{'span': (o, f)} for o, p, f in annotations[global_idx]['p_waves']]
            
            # Predict
            res = compute_instance_metrics(
                out['heatmap'][0:1].cpu().numpy(),
                out['width'][0:1].cpu().numpy(),
                target_spans
            )
            
            # Accumulate scores for global metric
            all_tp_lists.append(res['tp_list'])
            all_scores.append(res['scores'])
            total_gt += res['n_gt']
            total_tp += res['tp']
            total_fp += res['fp']
            total_fn += res['fn']
            record_results.append(res)
            
    m_ap, _, _ = calculate_mAP(all_tp_lists, all_scores, total_gt)
    avg_prec = np.mean([r['precision'] for r in record_results])
    avg_rec = np.mean([r['recall'] for r in record_results])
    avg_f1 = np.mean([r['f1'] for r in record_results])

    print("\n=====================================================")
    print(" FINAL TEST SET RESULTS (Real LUDB Data)")
    print("=====================================================")
    print(f"  Precision : {avg_prec:.4f}")
    print(f"  Recall    : {avg_rec:.4f}")
    print(f"  F1 Score  : {avg_f1:.4f}")
    print(f"  mAP@0.5   : {m_ap:.4f}")
    print("-----------------------------------------------------")
    print(f"  True Positives  (TP): {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    print("=====================================================\n")

    # 4. Confusion Matrix Generation
    print("📊 Generating Visual Confusion Matrix...")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(total_tp, total_fp, total_fn, save_path=cm_path)
    print(f"✅ Confusion Matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()

"""
Main Benchmarking Runner
========================
This script is the CORE of the benchmarking suite. It:
  1. Loads the LUDB dataset using the same pipeline as train.py
  2. Trains the SimpleCNN Baseline on the same data split
  3. Trains the 1D U-Net Baseline on the same data split
  4. Evaluates the pre-trained AtrionNet Hybrid on the held-out test set
  5. Saves all results to a CSV file for documentation

HOW TO RUN:
    cd ml_component
    python benchmarking/01_benchmark_runner.py

OUTPUT:
    benchmarking/results/all_results.csv        ← Comparison table
    benchmarking/results/per_record_results.csv ← Per-patient breakdown
"""

import sys
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Path Setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import DATA_DIR, CHECKPOINT_DIR
from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.modeling.atrion_net import AtrionNetHybrid
from src.losses.segmentation_losses import create_instance_loss
from src.engine.atrion_evaluator import compute_instance_metrics, calculate_mAP

from benchmarking.baselines.simple_cnn_baseline import SimpleCNNBaseline
from benchmarking.baselines.unet_1d_baseline import UNet1D

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS      = 80          # Baselines use fewer epochs — they are simpler
BATCH_SIZE  = 16
LR          = 1e-4
SEED        = 42          # MUST match train.py for identical test split
CONF_THRESH = 0.35
NMS_DIST    = 80
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_dataset():
    print("\n📁 Loading LUDB Dataset...")
    loader = LUDBLoader(DATA_DIR)
    signals, annotations = loader.get_all_data()
    total = len(signals)
    indices = np.random.permutation(total)
    tr = int(total * 0.70)
    val = int(total * 0.85)
    idx_train = indices[:tr]
    idx_val   = indices[tr:val]
    idx_test  = indices[val:]
    print(f"   Split: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
    return signals, annotations, idx_train, idx_val, idx_test


def make_loaders(signals, annotations, idx_train, idx_val, idx_test):
    train_ds = AtrionInstanceDataset([signals[i] for i in idx_train],
                                     [annotations[i] for i in idx_train], is_train=True)
    val_ds   = AtrionInstanceDataset([signals[i] for i in idx_val],
                                     [annotations[i] for i in idx_val],   is_train=False)
    test_ds  = AtrionInstanceDataset([signals[i] for i in idx_test],
                                     [annotations[i] for i in idx_test],  is_train=False)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False),
            DataLoader(test_ds,  batch_size=1,          shuffle=False))


# ── Training Function ─────────────────────────────────────────────────────────
def train_model(model, train_loader, model_name):
    print(f"\n🚀 Training [{model_name}] for {EPOCHS} epochs on {DEVICE}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.to(DEVICE)
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            sig = batch['signal'].to(DEVICE)
            tgt = {k: v.to(DEVICE) for k, v in batch.items() if k != 'signal'}
            optimizer.zero_grad()
            out = model(sig)
            loss = create_instance_loss(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} — Loss: {ep_loss/len(train_loader):.4f}")
    elapsed = time.time() - start
    print(f"   ✅ Training complete in {elapsed/60:.1f} minutes")
    return model, elapsed


# ── Evaluation Function ───────────────────────────────────────────────────────
def evaluate_model(model, test_loader, annotations, idx_test, model_name):
    print(f"\n📊 Evaluating [{model_name}] on test set...")
    model.eval()
    all_prec, all_rec, all_f1 = [], [], []
    all_scores, all_labels    = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sig = batch['signal'].to(DEVICE)
            out = model(sig)

            heatmap   = out['heatmap'][0, 0].cpu().numpy()
            width_map = out['width'][0, 0].cpu().numpy()

            global_idx    = idx_test[i]
            target_pwaves = annotations[global_idx]['p_waves']
            target_spans  = [{'span': (o, f), 'confidence': 1.0}
                             for (o, p, f) in target_pwaves]

            from src.engine.atrion_evaluator import get_instances_from_heatmap
            preds = get_instances_from_heatmap(
                heatmap, width_map,
                threshold=CONF_THRESH,
                distance=NMS_DIST,
                prominence=0.10
            )

            res = compute_instance_metrics(preds, target_spans, iou_threshold=0.5)
            all_prec.append(res['precision'])
            all_rec.append(res['recall'])
            all_f1.append(res['f1'])

            for p in preds:
                all_scores.append(p['confidence'])
                all_labels.append(1 if res['tp'] > 0 else 0)

    ap = calculate_mAP(all_scores, all_labels, sum(len(a['p_waves']) for a in
                                                   [annotations[i] for i in idx_test]))
    return {
        'model':     model_name,
        'precision': round(np.mean(all_prec), 4),
        'recall':    round(np.mean(all_rec),  4),
        'f1':        round(np.mean(all_f1),   4),
        'f1_std':    round(np.std(all_f1),    4),
        'mAP@0.5':   round(ap,                4),
    }, all_f1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    signals, annotations, idx_train, idx_val, idx_test = load_dataset()
    train_loader, val_loader, test_loader = make_loaders(
        signals, annotations, idx_train, idx_val, idx_test)

    all_results    = []
    all_f1_per_rec = {}

    # ── Baseline 1: Simple CNN ────────────────────────────────────────────────
    cnn = SimpleCNNBaseline(in_channels=12)
    cnn, cnn_time = train_model(cnn, train_loader, 'SimpleCNN_Baseline')
    cnn_res, cnn_f1s = evaluate_model(cnn, test_loader, annotations, idx_test, 'SimpleCNN_Baseline')
    cnn_res['train_time_min'] = round(cnn_time / 60, 1)
    all_results.append(cnn_res)
    all_f1_per_rec['SimpleCNN_Baseline'] = cnn_f1s

    # ── Baseline 2: 1D U-Net ─────────────────────────────────────────────────
    unet = UNet1D(in_channels=12)
    unet, unet_time = train_model(unet, train_loader, 'UNet1D_Baseline')
    unet_res, unet_f1s = evaluate_model(unet, test_loader, annotations, idx_test, 'UNet1D_Baseline')
    unet_res['train_time_min'] = round(unet_time / 60, 1)
    all_results.append(unet_res)
    all_f1_per_rec['UNet1D_Baseline'] = unet_f1s

    # ── Proposed: AtrionNet Hybrid (load pre-trained weights) ────────────────
    atrion_path = os.path.join(CHECKPOINT_DIR, 'atrion_hybrid_best.pth')
    if os.path.exists(atrion_path):
        atrion = AtrionNetHybrid(in_channels=12).to(DEVICE)
        atrion.load_state_dict(torch.load(atrion_path, map_location=DEVICE))
        atrion_res, atrion_f1s = evaluate_model(
            atrion, test_loader, annotations, idx_test, 'AtrionNet_Hybrid')
        atrion_res['train_time_min'] = 'Pre-trained'
        all_results.append(atrion_res)
        all_f1_per_rec['AtrionNet_Hybrid'] = atrion_f1s
    else:
        print(f"\n⚠️  AtrionNet weights not found at {atrion_path}. Run train.py first.")

    # ── Save Results to CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'all_results.csv')
    fieldnames = ['model', 'precision', 'recall', 'f1', 'f1_std', 'mAP@0.5', 'train_time_min']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✅ Results saved to: {csv_path}")

    # ── Print Final Comparison Table ───────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'F1 Std':>10} {'mAP@0.5':>10}")
    print("=" * 75)
    for r in all_results:
        print(f"{r['model']:<25} {r['precision']:>10} {r['recall']:>10} "
              f"{r['f1']:>10} {r['f1_std']:>10} {r['mAP@0.5']:>10}")
    print("=" * 75)

    # ── Save per-record CSV ────────────────────────────────────────────────────
    per_record_path = os.path.join(RESULTS_DIR, 'per_record_results.csv')
    max_len = max(len(v) for v in all_f1_per_rec.values())
    with open(per_record_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Record_Index'] + list(all_f1_per_rec.keys()))
        for i in range(max_len):
            row = [i]
            for model_name in all_f1_per_rec:
                vals = all_f1_per_rec[model_name]
                row.append(round(vals[i], 4) if i < len(vals) else '')
            writer.writerow(row)
    print(f"✅ Per-record results saved to: {per_record_path}")


if __name__ == '__main__':
    main()

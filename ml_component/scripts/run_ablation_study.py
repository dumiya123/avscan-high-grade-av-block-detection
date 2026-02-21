"""
Ablation Study: AtrionNet Baseline vs Hybrid.
Automatically compared CNN-only vs CNN+BiLSTM to validate research improvements.
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from src.modeling.atrion_net import AtrionNetHybrid, AtrionNetBaseline
from src.data_pipeline.ludb_loader import LUDBLoader
from src.data_pipeline.instance_dataset import AtrionInstanceDataset
from src.losses.segmentation_losses import create_instance_loss

def run_experiment(model_class, name, signals, annotations, indices, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_idx, val_idx = indices[:int(0.8*len(indices))], indices[int(0.8*len(indices)):]
    
    train_ds = AtrionInstanceDataset(signals[tr_idx], [annotations[i] for i in tr_idx], is_train=True)
    val_ds = AtrionInstanceDataset(signals[val_idx], [annotations[i] for i in val_idx], is_train=False)
    
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    v_loader = DataLoader(val_ds, batch_size=16)
    
    model = model_class().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    history = []
    print(f"🧪 Running Experiment: {name}")
    for e in range(epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            l = create_instance_loss(model(b['signal'].to(device)), {k:v.to(device) for k,v in b.items() if k!='signal'})
            l.backward(); opt.step()
        
        model.eval(); v_l = 0
        with torch.no_grad():
            for b in v_loader:
                v_l += create_instance_loss(model(b['signal'].to(device)), {k:v.to(device) for k,v in b.items() if k!='signal'}).item()
        history.append(v_l/len(v_loader))
        print(f"  Epoch {e+1}: Val Loss = {history[-1]:.4f}")
        
    return history

def main():
    loader = LUDBLoader("data/raw/ludb")
    sigs, anns = loader.get_all_data()
    indices = np.arange(len(sigs))
    np.random.shuffle(indices)
    
    # Run ablation
    baseline_hist = run_experiment(AtrionNetBaseline, "Baseline (CNN)", sigs, anns, indices)
    hybrid_hist = run_experiment(AtrionNetHybrid, "Hybrid (CNN+BiLSTM)", sigs, anns, indices)
    
    # Save results
    os.makedirs("reports/plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_hist, label='CNN Baseline', marker='o', linewidth=2)
    plt.plot(hybrid_hist, label='CNN+BiLSTM Hybrid (AtrionNet)', marker='s', linewidth=2)
    plt.title("Ablation Study: Impact of Temporal Modeling on P-Wave Detection")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Multi-Task Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/plots/ablation_study_comparison.png", dpi=300)
    
    print("\n✅ Ablation Study Complete. Plot saved to reports/plots/ablation_study_comparison.png")

if __name__ == "__main__":
    import os
    main()

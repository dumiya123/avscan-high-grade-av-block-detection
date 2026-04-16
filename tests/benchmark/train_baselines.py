"""
BASELINE TRAINER — Fair Apples-to-Apples Comparison
[Sup #3, #6]

Trains AtrionNetBaseline (1D U-Net) and CNN-LSTM on the EXACT same:
  - Dataset      : LUDB (or synthetic fallback)
  - Split        : seed=42, 70/15/15 patient-level permutation
  - Optimizer    : AdamW, lr=1e-4, weight_decay=1e-4
  - Scheduler    : ReduceLROnPlateau (mode='max', factor=0.5, patience=20)
  - Loss         : create_instance_loss (focal + smooth_l1 + BCE+Dice)
  - Epochs       : 150 max (patience=25 early stopping)
  - Augmentation : Same Joung-style augmentations (is_train=True)

Saved weights:
  tests/outputs/baseline_weights/unet_baseline_best.pth
  tests/outputs/baseline_weights/cnn_lstm_best.pth

CLI:
  cd AtrionNet_Implementation
  # Train both baselines (recommended before Phase 2)
  python tests/benchmark/train_baselines.py

  # Train only one
  python tests/benchmark/train_baselines.py --model unet
  python tests/benchmark/train_baselines.py --model cnn_lstm

  # Reduce epochs for quick test (e.g. 10 epochs)
  python tests/benchmark/train_baselines.py --epochs 10

[CLINICAL-VAL] All logs prefixed with [CLINICAL-VAL].
[Sup #3, #6]
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────────
import os
import sys
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ── path bootstrap ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT      = PROJECT_ROOT / "ml_component"
BASELINE_WEIGHTS_DIR = PROJECT_ROOT / "tests" / "outputs" / "baseline_weights"
BASELINE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[CLINICAL-VAL] %(levelname)s — %(message)s"
)
log = logging.getLogger("baseline_trainer")

# ── Identical hyperparameters to train.py ATRION_CONFIG ──────────────────────
BASELINE_CONFIG = {
    "EPOCHS":        150,
    "PATIENCE":      25,
    "BATCH_SIZE":    16,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY":  1e-4,
    # Loss weights — identical to AtrionNet
    "FOCAL_ALPHA":   2.0,
    "FOCAL_BETA":    4.0,
    "LOSS_WEIGHT_HM": 2.0,
    "LOSS_WEIGHT_W":  1.0,
    "LOSS_WEIGHT_M":  1.0,
    # Evaluation thresholds — identical to AtrionNet
    "EVAL_CONF":  0.45,
    "EVAL_DIST":  60,
    "EVAL_PROM":  0.10,
    # Training
    "ACCUMULATION_STEPS": 4,
    "SEED": 42,
    "N_LEADS": 12,
    "SEQ_LEN": 5000,
}


# ════════════════════════════════════════════════════════════════════════════
# § 0 — Determinism
# ════════════════════════════════════════════════════════════════════════════
def _seed_all(seed: int = BASELINE_CONFIG["SEED"]) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


_seed_all()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"[CLINICAL-VAL] Baseline Trainer — Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# § 1 — CNN-LSTM Baseline Definition
#        (Identical to compare_baselines.py — defined here independently
#         so train_baselines.py has zero external test/ imports)
# ════════════════════════════════════════════════════════════════════════════
class CNNLSTM(nn.Module):
    """
    Lightweight 1D CNN encoder + BiLSTM + transposed decoder.
    Same output schema as AtrionNetHybrid: {heatmap, width, mask}.
    [Sup #3, #6]
    """
    name = "CNN-LSTM"

    def __init__(self, in_channels: int = 12, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=7, padding=3), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(256, hidden, batch_first=True,
                            bidirectional=True, num_layers=1)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(hidden * 2, 128, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose1d(128, 64,  kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose1d(64,  32,  kernel_size=2, stride=2), nn.ReLU(),
        )
        self.heatmap_head = nn.Sequential(nn.Conv1d(32, 1, 1), nn.Sigmoid())
        self.width_head   = nn.Conv1d(32, 1, 1)
        self.mask_head    = nn.Conv1d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat  = self.enc(x)
        out_t, _ = self.lstm(feat.permute(0, 2, 1))
        dec   = self.up(out_t.permute(0, 2, 1))
        return {
            "heatmap": self.heatmap_head(dec),
            "width":   self.width_head(dec),
            "mask":    self.mask_head(dec),
        }


# ════════════════════════════════════════════════════════════════════════════
# § 2 — Data Loading (exact mirror of train.py)
# ════════════════════════════════════════════════════════════════════════════
def _load_splits() -> Tuple:
    """
    Replicates the EXACT split logic from train.py lines 68-96.
    Returns train_loader, val_loader, idx_val, annotations.
    """
    from src.data_pipeline.ludb_loader import LUDBLoader
    from src.data_pipeline.instance_dataset import AtrionInstanceDataset

    DATA_DIR = ML_ROOT / "data" / "raw" / "ludb"
    log.info(f"[CLINICAL-VAL] Loading LUDB from {DATA_DIR}")
    loader = LUDBLoader(str(DATA_DIR))
    signals, annotations = loader.get_all_data()

    # ── EXACT seed sequence from train.py line 68-73 ─────────────────────
    np.random.seed(BASELINE_CONFIG["SEED"])
    torch.manual_seed(BASELINE_CONFIG["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASELINE_CONFIG["SEED"])

    total     = len(signals)
    indices   = np.random.permutation(total)
    tr_split  = int(total * 0.70)
    val_split = int(total * 0.85)

    idx_tr   = indices[:tr_split]
    idx_val  = indices[tr_split:val_split]

    log.info(f"[CLINICAL-VAL] Split → Train:{len(idx_tr)} "
             f"Val:{len(idx_val)} Test:{total - val_split}")

    train_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_tr],
        [annotations[i] for i in idx_tr],
        is_train=True   # Same Joung augmentations as AtrionNet
    )
    val_ds = AtrionInstanceDataset(
        [signals[i] for i in idx_val],
        [annotations[i] for i in idx_val],
        is_train=False
    )

    BS = BASELINE_CONFIG["BATCH_SIZE"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False,
                              num_workers=0, pin_memory=False)

    return train_loader, val_loader, idx_val, annotations


# ════════════════════════════════════════════════════════════════════════════
# § 3 — Evaluation Helper (inline AAMI — SOC safe)
# ════════════════════════════════════════════════════════════════════════════
def _quick_eval(model: nn.Module,
                val_loader: DataLoader,
                idx_val:    np.ndarray,
                annotations: List[Dict]) -> Tuple[float, float, float]:
    """
    Returns (micro_f1, micro_precision, micro_recall) for epoch logging.
    Uses same compute_instance_metrics from src — evaluator, not imported
    back into src, so SOC is maintained.
    """
    from src.engine.atrion_evaluator import compute_instance_metrics

    model.eval()
    total_tp = total_fp = total_fn = 0
    val_sample_idx = 0

    with torch.no_grad():
        for batch in val_loader:
            sig   = batch["signal"].to(DEVICE)
            out   = model(sig)
            for b_j in range(sig.size(0)):
                global_idx   = idx_val[val_sample_idx]
                target_spans = [{"span": (o, f)}
                                for o, p, f in annotations[global_idx]["p_waves"]]
                res = compute_instance_metrics(
                    out["heatmap"][b_j:b_j+1].cpu().numpy(),
                    out["width"][b_j:b_j+1].cpu().numpy(),
                    target_spans,
                    conf_threshold=BASELINE_CONFIG["EVAL_CONF"],
                    distance=BASELINE_CONFIG["EVAL_DIST"],
                    prominence=BASELINE_CONFIG["EVAL_PROM"],
                )
                total_tp += res["tp"]
                total_fp += res["fp"]
                total_fn += res["fn"]
                val_sample_idx += 1

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


# ════════════════════════════════════════════════════════════════════════════
# § 4 — Core Training Loop
# ════════════════════════════════════════════════════════════════════════════
def _train_baseline(model:        nn.Module,
                    model_name:   str,
                    save_path:    Path,
                    train_loader: DataLoader,
                    val_loader:   DataLoader,
                    idx_val:      np.ndarray,
                    annotations:  List[Dict],
                    max_epochs:   int = BASELINE_CONFIG["EPOCHS"]) -> None:
    """
    Identical training loop to train.py — same optimizer, scheduler,
    loss function, gradient accumulation, early stopping, and checkpointing.
    [Sup #3, #6]
    """
    from src.losses.segmentation_losses import create_instance_loss

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASELINE_CONFIG["LEARNING_RATE"],
        weight_decay=BASELINE_CONFIG["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )

    PATIENCE   = BASELINE_CONFIG["PATIENCE"]
    ACCUM      = BASELINE_CONFIG["ACCUMULATION_STEPS"]
    best_f1    = 0.0
    pat_ctr    = 0
    ckpt_path  = save_path.parent / f"{model_name}_checkpoint.pth"

    print(f"\n[CLINICAL-VAL] Training {model_name} — "
          f"max_epochs={max_epochs} patience={PATIENCE} device={DEVICE}")
    print(f"  Save → {save_path}\n")

    for epoch in range(max_epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for b_idx, batch in enumerate(tqdm(
                train_loader,
                desc=f"[{model_name}] Epoch {epoch+1}/{max_epochs} [Train]",
                leave=False)):
            sigs  = batch["signal"].to(DEVICE)
            targs = {k: v.to(DEVICE) for k, v in batch.items() if k != "signal"}

            out  = model(sigs)
            loss = create_instance_loss(
                out, targs,
                alpha=BASELINE_CONFIG["FOCAL_ALPHA"],
                beta=BASELINE_CONFIG["FOCAL_BETA"],
                hm_weight=BASELINE_CONFIG["LOSS_WEIGHT_HM"],
                w_weight=BASELINE_CONFIG["LOSS_WEIGHT_W"],
                m_weight=BASELINE_CONFIG["LOSS_WEIGHT_M"],
            )
            loss = loss / ACCUM
            loss.backward()

            if ((b_idx + 1) % ACCUM == 0) or (b_idx + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM

        train_loss = epoch_loss / len(train_loader)

        # ── Validation loss ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sig   = batch["signal"].to(DEVICE)
                targs = {k: v.to(DEVICE) for k, v in batch.items() if k != "signal"}
                out   = model(sig)
                val_loss += create_instance_loss(
                    out, targs,
                    alpha=BASELINE_CONFIG["FOCAL_ALPHA"],
                    beta=BASELINE_CONFIG["FOCAL_BETA"],
                    hm_weight=BASELINE_CONFIG["LOSS_WEIGHT_HM"],
                    w_weight=BASELINE_CONFIG["LOSS_WEIGHT_W"],
                    m_weight=BASELINE_CONFIG["LOSS_WEIGHT_M"],
                ).item()
        val_loss /= len(val_loader)

        # ── AAMI-based validation metrics ─────────────────────────────────
        val_f1, val_prec, val_rec = _quick_eval(
            model, val_loader, idx_val, annotations
        )

        scheduler.step(val_f1)

        print(f"  Epoch {epoch+1:03d} | TrLoss:{train_loss:.4f} | "
              f"VlLoss:{val_loss:.4f} | F1:{val_f1:.4f} | "
              f"P:{val_prec:.4f} R:{val_rec:.4f} | "
              f"LR:{optimizer.param_groups[0]['lr']:.2e}")

        # ── Checkpoint ───────────────────────────────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            pat_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ New best {model_name} F1={best_f1:.4f} — saved.")
        else:
            pat_ctr += 1

        # Save rolling checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1,
            "patience_counter": pat_ctr,
        }, ckpt_path)

        if pat_ctr >= PATIENCE:
            print(f"\n  ⏹ Early stopping at epoch {epoch+1} "
                  f"(best F1={best_f1:.4f})")
            break

    print(f"\n[CLINICAL-VAL] {model_name} training complete.")
    print(f"  Best Val F1 : {best_f1:.4f}")
    print(f"  Weights     : {save_path}\n")


# ════════════════════════════════════════════════════════════════════════════
# § 5 — Entry Point
# ════════════════════════════════════════════════════════════════════════════
def main(model_choice: str = "both", max_epochs: int = BASELINE_CONFIG["EPOCHS"]) -> None:
    _seed_all()

    print("\n" + "=" * 72)
    print("[CLINICAL-VAL] BASELINE TRAINER — Fair Apples-to-Apples [Sup #3, #6]")
    print(f"  Models    : {model_choice}")
    print(f"  Epochs    : {max_epochs} (patience={BASELINE_CONFIG['PATIENCE']})")
    print(f"  Loss      : focal + smooth_l1 + BCE+Dice  ← identical to AtrionNet")
    print(f"  Split     : seed=42, 70/15/15  ← identical to train.py")
    print(f"  Augments  : is_train=True  ← same Joung augmentations")
    print(f"  Output    : {BASELINE_WEIGHTS_DIR}")
    print("=" * 72 + "\n")

    # ── Load data splits ───────────────────────────────────────────────────
    train_loader, val_loader, idx_val, annotations = _load_splits()

    from src.modeling.atrion_net import AtrionNetBaseline

    # ── Train U-Net Baseline ───────────────────────────────────────────────
    if model_choice in ("unet", "both"):
        _seed_all()   # Reset seed before each model for isolation
        unet  = AtrionNetBaseline(in_channels=BASELINE_CONFIG["N_LEADS"])
        _train_baseline(
            model=unet,
            model_name="UNet-Baseline",
            save_path=BASELINE_WEIGHTS_DIR / "unet_baseline_best.pth",
            train_loader=train_loader,
            val_loader=val_loader,
            idx_val=idx_val,
            annotations=annotations,
            max_epochs=max_epochs,
        )

    # ── Train CNN-LSTM ─────────────────────────────────────────────────────
    if model_choice in ("cnn_lstm", "both"):
        _seed_all()   # Reset seed before each model for isolation
        cnn_lstm = CNNLSTM(in_channels=BASELINE_CONFIG["N_LEADS"])
        _train_baseline(
            model=cnn_lstm,
            model_name="CNN-LSTM",
            save_path=BASELINE_WEIGHTS_DIR / "cnn_lstm_best.pth",
            train_loader=train_loader,
            val_loader=val_loader,
            idx_val=idx_val,
            annotations=annotations,
            max_epochs=max_epochs,
        )

    print("\n" + "=" * 72)
    print("[CLINICAL-VAL] All baselines trained.")
    print(f"  Weights directory: {BASELINE_WEIGHTS_DIR}")
    print("  Next step: python tests/benchmark/compare_baselines.py")
    print("=" * 72 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# § 6 — CLI
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[CLINICAL-VAL] Train baselines for fair AtrionNet comparison"
    )
    parser.add_argument(
        "--model", choices=["unet", "cnn_lstm", "both"], default="both",
        help="Which baseline(s) to train (default: both)"
    )
    parser.add_argument(
        "--epochs", type=int, default=BASELINE_CONFIG["EPOCHS"],
        help=f"Max training epochs (default: {BASELINE_CONFIG['EPOCHS']})"
    )
    args = parser.parse_args()
    main(model_choice=args.model, max_epochs=args.epochs)

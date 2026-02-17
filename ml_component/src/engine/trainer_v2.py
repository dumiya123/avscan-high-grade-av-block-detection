"""
Training Engine for Multi-Label ECG Classification.

Features:
- Multi-label training with BCEWithLogitsLoss or Focal Loss
- Early stopping based on validation Macro F1
- Cosine annealing or ReduceOnPlateau LR scheduler
- Gradient clipping
- Reproducibility via seed control
- Checkpoint saving (best + last)
- Clean epoch-level logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Optional

from src.engine.losses_v2 import get_loss_function
from src.engine.evaluator_v2 import compute_all_metrics


def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, mode: str = 'max', min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict:
    """
    Train for one epoch.

    Returns:
        Dict with 'loss' (average epoch loss).
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for signals, labels in train_loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    return {'loss': avg_loss}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """
    Validate for one epoch.

    Returns:
        Dict with 'loss', 'macro_f1', 'mean_auroc', etc.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_probs = []
    all_labels = []

    for signals, labels in val_loader:
        signals = signals.to(device)
        labels = labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_all_metrics(y_true, y_prob, threshold)
    metrics['loss'] = avg_loss

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict,
    checkpoint_dir: str = 'checkpoints',
    experiment_name: str = 'experiment',
) -> Dict:
    """
    Full training loop with early stopping and best-model saving.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: torch.device.
        config: Training config dict (from experiment.py).
        checkpoint_dir: Where to save checkpoints.
        experiment_name: Name for checkpoint files.

    Returns:
        Dict with training history and best metrics.
    """
    # Setup
    epochs = config.get('epochs', 50)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    patience = config.get('early_stop_patience', 10)
    gradient_clip = config.get('gradient_clip', 1.0)
    scheduler_type = config.get('scheduler', 'cosine')

    ckpt_dir = Path(checkpoint_dir) / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
    else:
        scheduler = None

    # Loss function
    loss_type = config.get('loss_type', 'bce')
    pos_weight = config.get('pos_weight', None)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    criterion = get_loss_function(
        loss_type=loss_type,
        pos_weight=pos_weight,
        gamma=config.get('focal_gamma', 2.0),
        alpha=config.get('focal_alpha', 0.25),
    )

    # Early stopping
    early_stopper = EarlyStopping(patience=patience, mode='max')

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_macro_f1': [], 'val_auroc': [],
        'lr': [],
    }
    best_f1 = 0.0
    best_epoch = 0

    print(f"\n{'='*70}")
    print(f"  Training: {experiment_name}")
    print(f"  Epochs: {epochs} | LR: {lr} | Loss: {loss_type}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )

        # Validate
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        # LR scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler_type == 'cosine' and scheduler:
            scheduler.step()
        elif scheduler_type == 'plateau' and scheduler:
            scheduler.step(val_metrics['macro_f1'])

        elapsed = time.time() - t0

        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_auroc'].append(val_metrics['mean_auroc'])
        history['lr'].append(current_lr)

        # Print progress
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val AUROC: {val_metrics['mean_auroc']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save best model (based on Macro F1)
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_f1,
                'val_auroc': val_metrics['mean_auroc'],
                'config': config,
            }, ckpt_dir / 'best_model.pth')
            print(f"    ★ New best model saved (F1: {best_f1:.4f})")

        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_dir / 'last_model.pth')

        # Early stopping
        if early_stopper(val_metrics['macro_f1']):
            print(f"\n  ⚠ Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Save training history
    with open(ckpt_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Training complete!")
    print(f"  Best Macro F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"  Checkpoints saved to: {ckpt_dir}")

    return {
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'history': history,
        'checkpoint_dir': str(ckpt_dir),
    }

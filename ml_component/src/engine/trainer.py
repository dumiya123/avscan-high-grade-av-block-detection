"""
Training script for ECG U-Net
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.ecg_unet import ECGUNet
from src.data_pipeline.loader import create_dataloaders
from src.data_pipeline.transforms import get_train_transforms, get_val_transforms
from src.engine.losses import MultiTaskLoss
from src.utils import set_seed, get_device, save_checkpoint, AverageMeter, format_time


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    seg_losses = AverageMeter()
    clf_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (signals, seg_masks, clf_labels) in enumerate(pbar):
        signals = signals.to(device)
        seg_masks = seg_masks.to(device)
        clf_labels = clf_labels.to(device)
        
        # Forward pass
        seg_pred, clf_pred = model(signals)
        
        # Calculate loss
        total_loss, seg_loss, clf_loss = criterion(seg_pred, seg_masks, clf_pred, clf_labels)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        losses.update(total_loss.item(), signals.size(0))
        seg_losses.update(seg_loss.item(), signals.size(0))
        clf_losses.update(clf_loss.item(), signals.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'seg': f'{seg_losses.avg:.4f}',
            'clf': f'{clf_losses.avg:.4f}'
        })
    
    return losses.avg, seg_losses.avg, clf_losses.avg


def validate(model, dataloader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    
    losses = AverageMeter()
    seg_losses = AverageMeter()
    clf_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for signals, seg_masks, clf_labels in pbar:
            signals = signals.to(device)
            seg_masks = seg_masks.to(device)
            clf_labels = clf_labels.to(device)
            
            # Forward pass
            seg_pred, clf_pred = model(signals)
            
            # Calculate loss
            total_loss, seg_loss, clf_loss = criterion(seg_pred, seg_masks, clf_pred, clf_labels)
            
            # Update metrics
            losses.update(total_loss.item(), signals.size(0))
            seg_losses.update(seg_loss.item(), signals.size(0))
            clf_losses.update(clf_loss.item(), signals.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'seg': f'{seg_losses.avg:.4f}',
                'clf': f'{clf_losses.avg:.4f}'
            })
    
    return losses.avg, seg_losses.avg, clf_losses.avg


def train_model(
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    seg_weight: float = 0.6,
    clf_weight: float = 0.4,
    resume: str = None,
    early_stop_patience: int = 15
):
    """
    Main training function
    
    Args:
        data_dir: Directory with processed data
        checkpoint_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        seg_weight: Weight for segmentation loss
        clf_weight: Weight for classification loss
        resume: Path to checkpoint to resume from
        early_stop_patience: Early stopping patience
    """
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Create directories
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir / f"run_{int(time.time())}")
    
    # Create dataloaders
    data_file = Path(data_dir) / "ecg_data.h5"
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("   Please run preprocessing first: python run.py preprocess")
        return
    
    print("📊 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_file,
        batch_size=batch_size,
        num_workers=4,
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms()
    )
    
    # Create model
    print("🏗️  Creating model...")
    model = ECGUNet(
        in_channels=1,
        num_seg_classes=5,
        num_clf_classes=6,
        base_channels=64,
        use_attention=True
    ).to(device)
    
    # Print model summary
    from src.modeling.ecg_unet import model_summary
    model_summary(model)
    
    # Loss function
    criterion = MultiTaskLoss(
        seg_weight=seg_weight,
        clf_weight=clf_weight
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume:
        from src.utils import load_checkpoint
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("=" * 80)
    
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_seg_loss, train_clf_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_seg_loss, val_clf_loss = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_seg', train_seg_loss, epoch)
        writer.add_scalar('Loss/val_seg', val_seg_loss, epoch)
        writer.add_scalar('Loss/train_clf', train_clf_loss, epoch)
        writer.add_scalar('Loss/val_clf', val_clf_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{epochs-1} - {format_time(epoch_time)}")
        print(f"  Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Clf: {train_clf_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Clf: {val_clf_loss:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            {'seg_loss': val_seg_loss, 'clf_loss': val_clf_loss},
            checkpoint_path / f"checkpoint_epoch_{epoch}.pth",
            is_best=is_best
        )
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
            break
        
        print("=" * 80)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n✅ Training complete! Total time: {format_time(total_time)}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Best model saved to: {checkpoint_path / 'best_model.pth'}")
    
    writer.close()


if __name__ == "__main__":
    train_model()

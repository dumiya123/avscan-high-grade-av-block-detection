
import torch
import numpy as np
import random
import os
import time
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, metrics, path, is_best=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics
    }
    torch.save(state, path)
    
    if is_best:
        best_path = path.parent / 'best_model.pth'
        torch.save(state, best_path)
        # Also copy to last_model.pth for convenience
        last_path = path.parent / 'last_model.pth'
        torch.save(state, last_path)

def load_checkpoint(path, model, optimizer=None, device=None):
    if device is None:
        device = get_device()
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}"

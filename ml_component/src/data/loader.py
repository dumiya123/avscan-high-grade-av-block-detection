"""
PyTorch Dataset class for ECG data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from sklearn.model_selection import train_test_split


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG segmentation and classification
    
    Returns:
        signal: (1, seq_len) tensor
        seg_mask: (seq_len,) tensor with class labels 0-4
        av_block_label: scalar tensor with AV block class 0-5
    """
    
    def __init__(
        self,
        data_file: Path,
        indices: Optional[List[int]] = None,
        transform=None,
        target_length: int = 5000
    ):
        """
        Args:
            data_file: Path to HDF5 file
            indices: List of record indices to use (for train/val/test split)
            transform: Data augmentation transforms
            target_length: Target sequence length (pad/crop)
        """
        self.data_file = data_file
        self.transform = transform
        self.target_length = target_length
        
        # Load metadata
        with h5py.File(data_file, 'r') as hf:
            total_records = len(hf.keys())
            
            if indices is None:
                self.indices = list(range(total_records))
            else:
                self.indices = indices
            
            # Cache metadata
            self.metadata = []
            for idx in self.indices:
                record = hf[f'record_{idx}']
                self.metadata.append({
                    'av_block_label': record.attrs['av_block_label'],
                    'fs': record.attrs['fs'],
                    'record_name': record.attrs['record_name']
                })
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            signal: (1, seq_len) ECG signal
            seg_mask: (seq_len,) segmentation mask
            av_block_label: scalar AV block label
        """
        record_idx = self.indices[idx]
        
        # Load data from HDF5
        with h5py.File(self.data_file, 'r') as hf:
            record = hf[f'record_{record_idx}']
            signal = record['signal'][:]
            seg_mask = record['seg_mask'][:]
            av_block_label = record.attrs['av_block_label']
        
        # Pad or crop to target length
        signal, seg_mask = self._adjust_length(signal, seg_mask)
        
        # Convert to tensors
        signal = torch.from_numpy(signal).float().unsqueeze(0)  # (1, seq_len)
        seg_mask = torch.from_numpy(seg_mask).long()  # (seq_len,)
        av_block_label = torch.tensor(av_block_label, dtype=torch.long)
        
        # Apply transforms
        if self.transform is not None:
            signal, seg_mask = self.transform(signal, seg_mask)
        
        return signal, seg_mask, av_block_label
    
    def _adjust_length(
        self,
        signal: np.ndarray,
        seg_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or crop signal and mask to target length
        
        Args:
            signal: ECG signal
            seg_mask: Segmentation mask
            
        Returns:
            Adjusted signal and mask
        """
        current_length = len(signal)
        
        if current_length < self.target_length:
            # Pad
            pad_length = self.target_length - current_length
            signal = np.pad(signal, (0, pad_length), mode='constant')
            seg_mask = np.pad(seg_mask, (0, pad_length), mode='constant')
        elif current_length > self.target_length:
            # Crop (take center portion)
            start = (current_length - self.target_length) // 2
            signal = signal[start:start + self.target_length]
            seg_mask = seg_mask[start:start + self.target_length]
        
        return signal, seg_mask


def create_data_splits(
    data_file: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits with stratification by AV block label
    
    Args:
        data_file: Path to HDF5 file
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Load all labels
    with h5py.File(data_file, 'r') as hf:
        num_records = len(hf.keys())
        labels = []
        for i in range(num_records):
            labels.append(hf[f'record_{i}'].attrs['av_block_label'])
    
    indices = np.arange(num_records)
    
    # First split: train vs (val + test)
    try:
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=random_state
        )
    except ValueError:
        print("⚠️ Warning: Could not stratify first split. Falling back to random split.")
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            stratify=None,
            random_state=random_state
        )
    
    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_indices]
    val_size = val_ratio / (val_ratio + test_ratio)
    
    try:
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_size),
            stratify=temp_labels,
            random_state=random_state
        )
    except ValueError:
        print("⚠️ Warning: Could not stratify second split. Falling back to random split.")
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_size),
            stratify=None,
            random_state=random_state
        )
    
    print(f"📊 Data split:")
    print(f"   Train: {len(train_indices)} samples ({train_ratio:.1%})")
    print(f"   Val:   {len(val_indices)} samples ({val_ratio:.1%})")
    print(f"   Test:  {len(test_indices)} samples ({test_ratio:.1%})")
    
    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()


def create_dataloaders(
    data_file: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders
    
    Args:
        data_file: Path to HDF5 file
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_transform: Augmentation for training
        val_transform: Transform for validation/test
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create splits
    train_idx, val_idx, test_idx = create_data_splits(data_file)
    
    # Create datasets
    train_dataset = ECGDataset(data_file, indices=train_idx, transform=train_transform)
    val_dataset = ECGDataset(data_file, indices=val_idx, transform=val_transform)
    test_dataset = ECGDataset(data_file, indices=test_idx, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    data_file = Path("data/processed/ecg_data.h5")
    
    if data_file.exists():
        print("Testing dataset loading...")
        dataset = ECGDataset(data_file)
        print(f"Dataset size: {len(dataset)}")
        
        # Load one sample
        signal, seg_mask, av_label = dataset[0]
        print(f"Signal shape: {signal.shape}")
        print(f"Mask shape: {seg_mask.shape}")
        print(f"AV block label: {av_label}")
    else:
        print(f"Data file not found: {data_file}")

"""
PTB-XL Dataset Handler.
Handles loading, label aggregation, and patient-wise fold splitting
following the official PTB-XL benchmark methodology.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import wfdb
import ast
from pathlib import Path
from typing import List, Optional, Tuple, Dict


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL 12-lead ECG multi-label classification.

    Key design decisions:
    - Uses official strat_fold for patient-wise stratification (no leakage).
    - Aggregates 71 SCP codes into 5 diagnostic superclasses.
    - Returns multi-hot labels (not single-label) for proper multi-label training.
    - Supports ECG-specific augmentations via a transform callable.
    """

    SUPERCLASS_MAP = {
        'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4
    }

    def __init__(
        self,
        data_dir: str,
        folds: List[int],
        sampling_rate: int = 500,
        transform=None,
        target_length: int = 5000,
    ):
        """
        Args:
            data_dir:      Path to PTB-XL root (contains ptbxl_database.csv).
            folds:         Which strat_folds to include (e.g. [1..8] for train).
            sampling_rate: 100 or 500 Hz.
            transform:     Optional augmentation callable(signal) -> signal.
            target_length: Pad/crop all signals to this length.
        """
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.target_length = target_length

        # ----------------------------------------------------------
        # 1. Load metadata
        # ----------------------------------------------------------
        db_path = self.data_dir / 'ptbxl_database.csv'
        if not db_path.exists():
            raise FileNotFoundError(
                f"ptbxl_database.csv not found in {self.data_dir}. "
                f"Download PTB-XL first."
            )

        self.df = pd.read_csv(db_path, index_col='ecg_id')
        self.df.scp_codes = self.df.scp_codes.apply(ast.literal_eval)

        # ----------------------------------------------------------
        # 2. Load SCP statements for diagnostic aggregation
        # ----------------------------------------------------------
        scp_path = self.data_dir / 'scp_statements.csv'
        self.agg_df = pd.read_csv(scp_path, index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        # ----------------------------------------------------------
        # 3. Aggregate labels to superclasses
        # ----------------------------------------------------------
        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(
            self._aggregate_diagnostic
        )

        # ----------------------------------------------------------
        # 4. Filter by requested folds
        # ----------------------------------------------------------
        self.df = self.df[self.df.strat_fold.isin(folds)]

        # ----------------------------------------------------------
        # 5. Build multi-hot labels
        # ----------------------------------------------------------
        self.labels = np.zeros((len(self.df), len(self.SUPERCLASS_MAP)), dtype=np.float32)
        for i, (_, row) in enumerate(self.df.iterrows()):
            for cls_name in row['diagnostic_superclass']:
                if cls_name in self.SUPERCLASS_MAP:
                    self.labels[i, self.SUPERCLASS_MAP[cls_name]] = 1.0

        # ----------------------------------------------------------
        # 6. Choose waveform path column based on sampling rate
        # ----------------------------------------------------------
        if sampling_rate == 500:
            self.filename_col = 'filename_hr'
        else:
            self.filename_col = 'filename_lr'

        # Store indices for fast access
        self.indices = self.df.index.tolist()
        self.filenames = self.df[self.filename_col].values

        print(f"  PTBXLDataset: {len(self)} records from folds {folds}")
        self._print_class_distribution()

    # ----------------------------------------------------------
    # Label aggregation
    # ----------------------------------------------------------
    def _aggregate_diagnostic(self, scp_dict: dict) -> List[str]:
        """Map raw SCP codes to diagnostic superclasses."""
        superclasses = set()
        for scp_code, likelihood in scp_dict.items():
            if likelihood >= 50.0:  # confident annotation
                if scp_code in self.agg_df.index:
                    cls = self.agg_df.loc[scp_code].diagnostic_class
                    if isinstance(cls, str) and cls in self.SUPERCLASS_MAP:
                        superclasses.add(cls)
        return list(superclasses)

    def _print_class_distribution(self):
        """Print per-class sample counts."""
        counts = self.labels.sum(axis=0).astype(int)
        total = len(self.labels)
        print("  Class distribution:")
        for name, idx in self.SUPERCLASS_MAP.items():
            pct = counts[idx] / total * 100
            print(f"    {name:5s}: {counts[idx]:5d} ({pct:.1f}%)")

    # ----------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            signal: (12, target_length) float tensor
            label:  (5,) multi-hot float tensor
        """
        filename = self.filenames[idx]
        filepath = self.data_dir / filename

        # Load waveform using wfdb
        record = wfdb.rdrecord(str(filepath))
        signal = record.p_signal  # (seq_len, num_leads)

        # Transpose to (num_leads, seq_len)
        signal = signal.T.astype(np.float32)

        # Handle NaN/inf
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad or crop to target length
        signal = self._adjust_length(signal)

        # Per-lead z-score normalization
        for lead_idx in range(signal.shape[0]):
            lead = signal[lead_idx]
            std = lead.std()
            if std > 1e-6:
                signal[lead_idx] = (lead - lead.mean()) / std
            else:
                signal[lead_idx] = lead - lead.mean()

        # Apply augmentations
        if self.transform is not None:
            signal = self.transform(signal)

        # Convert to tensors
        signal_tensor = torch.from_numpy(signal).float()
        label_tensor = torch.from_numpy(self.labels[idx]).float()

        return signal_tensor, label_tensor

    def _adjust_length(self, signal: np.ndarray) -> np.ndarray:
        """Pad or crop signal to target_length."""
        _, current_length = signal.shape
        if current_length < self.target_length:
            pad_width = self.target_length - current_length
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
        elif current_length > self.target_length:
            start = (current_length - self.target_length) // 2
            signal = signal[:, start:start + self.target_length]
        return signal

    # ----------------------------------------------------------
    # Class weights (for imbalance handling)
    # ----------------------------------------------------------
    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for BCEWithLogitsLoss.
        Formula: weight_c = N / (C * count_c)
        """
        counts = self.labels.sum(axis=0)
        total = len(self.labels)
        num_classes = len(counts)

        weights = total / (num_classes * counts + 1e-6)
        weights = weights / weights.sum() * num_classes  # normalize

        return torch.from_numpy(weights).float()


def create_ptbxl_loaders(
    data_dir: str,
    batch_size: int = 64,
    sampling_rate: int = 500,
    num_workers: int = 4,
    train_transform=None,
    train_folds: List[int] = None,
    val_folds: List[int] = None,
    test_folds: List[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders using official PTB-XL folds.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if train_folds is None:
        train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    if val_folds is None:
        val_folds = [9]
    if test_folds is None:
        test_folds = [10]

    print("Creating PTB-XL DataLoaders...")

    train_ds = PTBXLDataset(
        data_dir, folds=train_folds, sampling_rate=sampling_rate,
        transform=train_transform,
    )
    val_ds = PTBXLDataset(
        data_dir, folds=val_folds, sampling_rate=sampling_rate,
    )
    test_ds = PTBXLDataset(
        data_dir, folds=test_folds, sampling_rate=sampling_rate,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader

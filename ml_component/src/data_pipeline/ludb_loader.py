"""
LUDB Data Loader for AtrionNet.
Specifically designed to extract P-wave instances (onset, peak, offset)
from the Lobachevsky University Database (LUDB).
"""

import os
import wfdb
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple

class LUDBLoader:
    def __init__(self, data_dir: str, sampling_rate: int = 500, target_length: int = 5000):
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.records = self._get_record_list()

    def _get_record_list(self) -> List[str]:
        """Finds all record names in the data directory recursively."""
        if not self.data_dir.exists():
            print(f"⚠️ Warning: Data directory {self.data_dir} does not exist.")
            return []
        
        # Search recursively for .hea files
        hea_files = list(self.data_dir.rglob("*.hea"))
        
        # Filter out files in special folders like .ipynb_checkpoints
        hea_files = [f for f in hea_files if ".ipynb_checkpoints" not in str(f)]
        
        # Store absolute paths to handle nested structures correctly
        self.record_paths = {f.stem: f.parent for f in hea_files}
        
        print(f"🔍 Found {len(hea_files)} .hea files in {self.data_dir}")
        return sorted(list(self.record_paths.keys()))

    def load_record(self, record_name: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Loads a single record and its P-wave annotations.
        """
        # Use the stored parent directory for this specific record
        record_parent = self.record_paths[record_name]
        record_path = str(record_parent / record_name)
        
        # 1. Load the 12-lead signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal.T # [12, seq_len]
        
        # 2. Resample/Crop to target length if necessary
        # Note: LUDB records are usually 10s at 500Hz = 5000 samples
        if signal.shape[1] > self.target_length:
            signal = signal[:, :self.target_length]
        elif signal.shape[1] < self.target_length:
            padding = self.target_length - signal.shape[1]
            signal = np.pad(signal, ((0,0), (0, padding)), mode='constant')

        # 3. Load Annotations (LUDB uses different annotators, we'll use Lead II annotations)
        # LUDB lead II annotations usually have the extension .i2 (lead i, ii, etc)
        # We look for P-wave components (P_onset, P_peak, P_offset)
        p_waves = []
        try:
            # Try to get labels for Lead II specifically (standard choice)
            ann = wfdb.rdann(record_path, 'ii')
            
            # LUDB uses specific characters for onsets/offsets
            # '(' : onset, 'p' : peak, ')' : offset
            # We need to find sequences of ( p )
            
            i = 0
            while i < len(ann.sample):
                if ann.symbol[i] == '(' and i+2 < len(ann.sample):
                    if ann.symbol[i+1] == 'p' and ann.symbol[i+2] == ')':
                        onset = ann.sample[i]
                        peak = ann.sample[i+1]
                        offset = ann.sample[i+2]
                        
                        # Only keep if within our target length
                        if offset < self.target_length:
                            p_waves.append((onset, peak, offset))
                        i += 3
                    else:
                        i += 1
                else:
                    i += 1
        except Exception as e:
            # If Lead II annotation is missing, skip or try another lead
            pass

        return signal, p_waves

    def get_all_data(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Loads all records in the directory.
        """
        all_signals = []
        all_annotations = []
        
        print(f"📂 Loading {len(self.records)} LUDB records...")
        for r_name in self.records:
            sig, p_waves = self.load_record(r_name)
            if len(p_waves) > 0: # Only keep records with P-wave labels
                all_signals.append(sig)
                all_annotations.append({'p_waves': p_waves})
        
        return np.array(all_signals), all_annotations

if __name__ == "__main__":
    # Test loader with a dummy path
    loader = LUDBLoader("./data/raw/ludb")
    print(f"Found {len(loader.records)} records.")

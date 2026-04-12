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

        # 3. Load Annotations
        p_waves = []
        annotation_found = False
        leads_to_try = ['ii', 'v1', 'v5', 'i', 'iii', 'avf']
        
        for lead in leads_to_try:
            try:
                ann = wfdb.rdann(record_path, lead)
                annotation_found = True
                i = 0
                while i < len(ann.sample):
                    if ann.symbol[i] == '(' and i+2 < len(ann.sample):
                        if ann.symbol[i+1] == 'p' and ann.symbol[i+2] == ')':
                            onset = ann.sample[i]
                            peak = ann.sample[i+1]
                            offset = ann.sample[i+2]
                            if offset < self.target_length:
                                p_waves.append((onset, peak, offset))
                            i += 3
                        else:
                            i += 1
                    else:
                        i += 1
                if annotation_found:
                    break
            except Exception:
                continue

        if not annotation_found:
            print(f"⚠️ Warning: No valid annotation file found for {record_name} in standard leads.")

        return signal, p_waves, annotation_found

    def get_all_data(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Loads all records in the directory.
        """
        all_signals = []
        all_annotations = []
        
        print(f"📂 Loading {len(self.records)} LUDB records...")
        for r_name in self.records:
            sig, p_waves, ann_found = self.load_record(r_name)
            if ann_found: # Keep records even if len(p_waves) == 0, as long as it was annotated
                all_signals.append(sig)
                all_annotations.append({'p_waves': p_waves})
        
        return np.array(all_signals), all_annotations

if __name__ == "__main__":
    # Test loader with a dummy path
    loader = LUDBLoader("./data/raw/ludb")
    print(f"Found {len(loader.records)} records.")

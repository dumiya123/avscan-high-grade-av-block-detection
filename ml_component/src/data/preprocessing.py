"""
ECG preprocessing module with 5-class segmentation label generation
"""

import wfdb
import numpy as np
from scipy import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
from tqdm import tqdm
import pandas as pd


def bandpass_filter(ecg_signal: np.ndarray, lowcut: float = 0.5, 
                    highcut: float = 40.0, fs: int = 500, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to ECG signal
    
    Args:
        ecg_signal: Raw ECG signal
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered ECG signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    
    return filtered_signal


def normalize_signal(ecg_signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalization (zero mean, unit variance)
    
    Args:
        ecg_signal: ECG signal
        
    Returns:
        Normalized signal
    """
    mean = np.mean(ecg_signal)
    std = np.std(ecg_signal)
    
    if std == 0:
        return ecg_signal - mean
    
    return (ecg_signal - mean) / std


def load_ecg_record(record_path: Path, channel: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    Load ECG record using wfdb
    
    Args:
        record_path: Path to record (without extension)
        channel: Channel index to load
        
    Returns:
        Tuple of (signal, metadata)
    """
    try:
        record = wfdb.rdrecord(str(record_path))
        # LUDB uses .i extension for annotations (reference)
        try:
            annotation = wfdb.rdann(str(record_path), 'i')
        except:
            # Fallback to .atr if .i fails (for compatibility)
            annotation = wfdb.rdann(str(record_path), 'atr')
        
        # Extract signal
        if record.n_sig > 1:
            ecg_signal = record.p_signal[:, channel]
        else:
            ecg_signal = record.p_signal.flatten()
        
        metadata = {
            'fs': record.fs,
            'sig_len': record.sig_len,
            'units': record.units,
            'annotations': annotation
        }
        
        return ecg_signal, metadata
    
    except Exception as e:
        print(f"❌ Error loading {record_path}: {e}")
        return None, None


def extract_wave_annotations(annotation: wfdb.Annotation, signal_length: int) -> Dict[str, List[int]]:
    """
    Extract P, QRS, T wave locations from annotations
    
    Args:
        annotation: wfdb Annotation object
        signal_length: Length of signal
        
    Returns:
        Dictionary with wave type and sample indices
    """
    waves = {
        'P': [],
        'QRS': [],
        'T': []
    }
    
    # Map annotation symbols to wave types
    # wfdb annotation symbols: '(' = P-wave, 'N' = QRS, 't' = T-wave
    symbol_map = {
        '(': 'P',
        'p': 'P',
        'N': 'QRS',
        'R': 'QRS',
        'Q': 'QRS',
        'S': 'QRS',
        't': 'T',
        'T': 'T'
    }
    
    for i, symbol in enumerate(annotation.symbol):
        wave_type = symbol_map.get(symbol)
        if wave_type and annotation.sample[i] < signal_length:
            waves[wave_type].append(annotation.sample[i])
    
    return waves


def detect_dissociated_p_waves(
    p_indices: List[int],
    qrs_indices: List[int],
    fs: int = 500,
    min_pr: int = 50,  # ms
    max_pr: int = 300  # ms
) -> Tuple[List[int], List[int]]:
    """
    Classify P-waves as associated or dissociated based on PR intervals
    
    Args:
        p_indices: List of P-wave sample indices
        qrs_indices: List of QRS sample indices
        fs: Sampling frequency
        min_pr: Minimum PR interval (ms)
        max_pr: Maximum PR interval (ms)
        
    Returns:
        Tuple of (associated_p_indices, dissociated_p_indices)
    """
    min_pr_samples = int((min_pr / 1000) * fs)
    max_pr_samples = int((max_pr / 1000) * fs)
    
    associated = []
    dissociated = []
    
    for p_idx in p_indices:
        # Find next QRS after this P-wave
        following_qrs = [q for q in qrs_indices if q > p_idx]
        
        if not following_qrs:
            # No QRS after this P-wave → dissociated
            dissociated.append(p_idx)
            continue
        
        next_qrs = following_qrs[0]
        pr_interval = next_qrs - p_idx
        
        if min_pr_samples <= pr_interval <= max_pr_samples:
            # Normal PR interval → associated
            associated.append(p_idx)
        else:
            # Abnormal PR interval → dissociated
            dissociated.append(p_idx)
    
    return associated, dissociated


def create_segmentation_mask(
    signal_length: int,
    waves: Dict[str, List[int]],
    window_size: int = 50
) -> np.ndarray:
    """
    Create 5-class segmentation mask from wave annotations
    
    Classes:
        0: Background
        1: P-wave (associated)
        2: P-wave (dissociated)
        3: QRS complex
        4: T-wave
    
    Args:
        signal_length: Length of ECG signal
        waves: Dictionary with 'P_associated', 'P_dissociated', 'QRS', 'T' indices
        window_size: Window size around each annotation (samples)
        
    Returns:
        Segmentation mask array
    """
    mask = np.zeros(signal_length, dtype=np.int64)
    
    # Class mapping
    class_map = {
        'P_associated': 1,
        'P_dissociated': 2,
        'QRS': 3,
        'T': 4
    }
    
    # Different window sizes for different waves
    wave_windows = {
        'P_associated': 40,  # P-wave is ~80ms
        'P_dissociated': 40,
        'QRS': 50,           # QRS is ~100ms
        'T': 100             # T-wave is ~200ms
    }
    
    # Fill mask (order matters: background < P < QRS < T for priority)
    for wave_type, class_id in class_map.items():
        if wave_type in waves:
            window = wave_windows.get(wave_type, window_size)
            for idx in waves[wave_type]:
                start = max(0, idx - window // 2)
                end = min(signal_length, idx + window // 2)
                mask[start:end] = class_id
    
    return mask


def assign_av_block_label(
    p_associated: List[int],
    p_dissociated: List[int],
    qrs_indices: List[int],
    fs: int = 500
) -> int:
    """
    Assign AV block classification label based on P-wave and QRS patterns
    
    Labels:
        0: Normal sinus rhythm
        1: 1st degree AV block
        2: 2nd degree Type I (Wenckebach)
        3: 2nd degree Type II (Mobitz II)
        4: 3rd degree (complete heart block)
        5: VT with AV dissociation
    
    Args:
        p_associated: Associated P-wave indices
        p_dissociated: Dissociated P-wave indices
        qrs_indices: QRS indices
        fs: Sampling frequency
        
    Returns:
        AV block class label
    """
    total_p = len(p_associated) + len(p_dissociated)
    total_qrs = len(qrs_indices)
    
    if total_p == 0 or total_qrs == 0:
        return 0  # Default to normal
    
    p_qrs_ratio = total_p / total_qrs
    dissociation_ratio = len(p_dissociated) / total_p if total_p > 0 else 0
    
    # Calculate average RR interval
    if len(qrs_indices) > 1:
        rr_intervals = np.diff(qrs_indices)
        avg_rr = np.mean(rr_intervals) / fs * 1000  # ms
        rr_variability = np.std(rr_intervals) / fs * 1000
    else:
        avg_rr = 0
        rr_variability = 0
    
    # Decision logic
    if dissociation_ratio > 0.7 and p_qrs_ratio > 1.5:
        # High dissociation + more P than QRS → 3rd degree
        return 4
    elif dissociation_ratio > 0.5:
        # Moderate dissociation → possible 2nd degree
        if rr_variability > 100:  # High RR variability
            return 2  # 2nd degree Type I (Wenckebach)
        else:
            return 3  # 2nd degree Type II
    elif len(p_associated) > 0 and len(qrs_indices) > 0:
        # Calculate average PR interval
        pr_intervals = []
        for p_idx in p_associated:
            following_qrs = [q for q in qrs_indices if q > p_idx]
            if following_qrs:
                pr = (following_qrs[0] - p_idx) / fs * 1000
                pr_intervals.append(pr)
        
        if pr_intervals:
            avg_pr = np.mean(pr_intervals)
            if avg_pr > 200:  # Prolonged PR
                return 1  # 1st degree AV block
    
    return 0  # Normal sinus rhythm


def preprocess_ludb(raw_dir: Path, output_dir: Path) -> List[Dict]:
    """
    Preprocess LUDB dataset
    
    Args:
        raw_dir: Directory with raw LUDB data
        output_dir: Directory to save processed data
        
    Returns:
        List of processed record metadata
    """
    ludb_dir = raw_dir / "ludb"
    
    if not ludb_dir.exists():
        print(f"❌ LUDB directory not found: {ludb_dir}")
        return []
    
    # Get all records
    records_file = ludb_dir / "RECORDS"
    if records_file.exists():
        with open(records_file, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
    else:
        # Find all .dat files
        records = [f.stem for f in ludb_dir.glob("*.dat")]
    
    processed_data = []
    
    print(f"📊 Processing {len(records)} LUDB records...")
    
    for record_name in tqdm(records):
        record_path = ludb_dir / record_name
        
        # Load ECG
        ecg_signal, metadata = load_ecg_record(record_path)
        
        if ecg_signal is None:
            continue
        
        # Preprocess signal
        filtered = bandpass_filter(ecg_signal, fs=metadata['fs'])
        normalized = normalize_signal(filtered)
        
        # Extract annotations
        waves = extract_wave_annotations(metadata['annotations'], len(ecg_signal))
        
        # Detect dissociated P-waves
        p_associated, p_dissociated = detect_dissociated_p_waves(
            waves['P'], waves['QRS'], fs=metadata['fs']
        )
        
        # Create segmentation mask
        wave_dict = {
            'P_associated': p_associated,
            'P_dissociated': p_dissociated,
            'QRS': waves['QRS'],
            'T': waves['T']
        }
        seg_mask = create_segmentation_mask(len(ecg_signal), wave_dict)
        
        # Assign AV block label
        av_block_label = assign_av_block_label(
            p_associated, p_dissociated, waves['QRS'], fs=metadata['fs']
        )
        
        processed_data.append({
            'record_name': record_name,
            'signal': normalized,
            'seg_mask': seg_mask,
            'av_block_label': av_block_label,
            'fs': metadata['fs'],
            'dataset': 'ludb'
        })
    
    return processed_data


def preprocess_ptbxl(raw_dir: Path, output_dir: Path) -> List[Dict]:
    """
    Preprocess PTB-XL dataset
    
    Args:
        raw_dir: Directory with raw PTB-XL data
        output_dir: Directory to save processed data
        
    Returns:
        List of processed record metadata
    """
    ptbxl_dir = raw_dir / "ptbxl"
    
    if not ptbxl_dir.exists():
        print(f"❌ PTB-XL directory not found: {ptbxl_dir}")
        return []
    
    # Load metadata
    metadata_file = ptbxl_dir / "ptbxl_database.csv"
    if not metadata_file.exists():
        print(f"❌ PTB-XL metadata not found")
        return []
    
    df = pd.read_csv(metadata_file)
    
    # Filter for AV block cases (if available in diagnostic labels)
    # For now, process a subset
    processed_data = []
    
    print(f"📊 Processing PTB-XL records (subset)...")
    
    # Process first 100 records as example
    for idx, row in tqdm(df.head(100).iterrows(), total=100):
        # PTB-XL specific processing would go here
        # This is a placeholder - full implementation would load waveforms
        pass
    
    return processed_data


def save_processed_data(processed_data: List[Dict], output_file: Path):
    """
    Save processed data to HDF5 file
    
    Args:
        processed_data: List of processed records
        output_file: Output HDF5 file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as hf:
        for i, data in enumerate(processed_data):
            grp = hf.create_group(f"record_{i}")
            grp.create_dataset('signal', data=data['signal'], compression='gzip')
            grp.create_dataset('seg_mask', data=data['seg_mask'], compression='gzip')
            grp.attrs['av_block_label'] = data['av_block_label']
            grp.attrs['fs'] = data['fs']
            grp.attrs['record_name'] = data['record_name']
            grp.attrs['dataset'] = data['dataset']
    
    print(f"💾 Saved {len(processed_data)} records to {output_file}")


def preprocess_datasets(raw_dir: str = "data/raw", 
                       output_dir: str = "data/processed",
                       validate: bool = False):
    """
    Main preprocessing function for all datasets
    
    Args:
        raw_dir: Directory with raw data
        output_dir: Directory to save processed data
        validate: Whether to validate preprocessing
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    # Process LUDB
    ludb_data = preprocess_ludb(raw_path, output_path)
    
    # Process PTB-XL
    ptbxl_data = preprocess_ptbxl(raw_path, output_path)
    
    # Combine and save
    all_data = ludb_data + ptbxl_data
    
    if all_data:
        save_processed_data(all_data, output_path / "ecg_data.h5")
        
        if validate:
            print("\n🔍 Validating preprocessing...")
            validate_preprocessing(output_path / "ecg_data.h5")
    else:
        print("❌ No data processed!")


def validate_preprocessing(data_file: Path):
    """
    Validate preprocessed data
    
    Args:
        data_file: Path to HDF5 file
    """
    with h5py.File(data_file, 'r') as hf:
        num_records = len(hf.keys())
        print(f"✅ Total records: {num_records}")
        
        # Check first record
        if num_records > 0:
            first_record = hf['record_0']
            print(f"   Signal shape: {first_record['signal'].shape}")
            print(f"   Mask shape: {first_record['seg_mask'].shape}")
            print(f"   AV block label: {first_record.attrs['av_block_label']}")
            print(f"   Sampling rate: {first_record.attrs['fs']} Hz")
            
            # Check class distribution
            mask = first_record['seg_mask'][:]
            unique, counts = np.unique(mask, return_counts=True)
            print(f"   Class distribution: {dict(zip(unique, counts))}")


if __name__ == "__main__":
    preprocess_datasets(validate=True)

import os
import wfdb
import numpy as np
from pathlib import Path

def extract_record(record_id):
    data_dir = Path("f:/Final_Year/Final_Semester_one/Final_Year_Research_Project/AtrionNet_Implementation/ml_component/data/raw/ludb/data")
    save_path = Path(f"f:/Final_Year/Final_Semester_one/Final_Year_Research_Project/AtrionNet_Implementation/test_case_{record_id}.npy")
    
    print(f"Reading LUDB Record {record_id}...")
    
    try:
        # Read the record
        record = wfdb.rdrecord(str(data_dir / record_id))
        
        # Signal is in (samples, channels) -> (5000, 12)
        # Dashboard expects (channels, samples) -> (12, 5000)
        signal = record.p_signal.T
        
        # Save as npy
        np.save(save_path, signal)
        print(f"SUCCESS: Saved as {save_path}")
        print(f"Diagnosis from .hea: III degree AV-block (Complete Heart Block)")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    extract_record("34")

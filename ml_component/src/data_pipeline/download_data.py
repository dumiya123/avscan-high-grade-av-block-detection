"""
Official LUDB Data Downloader (AtrionNet Research)
==================================================
Downloads the 200 records from PhysioNet for training.
"""

import os
import wfdb
import tqdm

def download_dataset():
    target_dir = os.path.join("data", "raw", "ludb")
    os.makedirs(target_dir, exist_ok=True)

    print(f"STATUS: Downloading records to {target_dir}...")
    
    # Download the record list
    try:
        record_list = wfdb.get_record_list('ludb')
        print(f"INFO: Found {len(record_list)} records in LUDB.")
        
        for record_name in tqdm.tqdm(record_list):
            # Download signal and header
            wfdb.dl_database('ludb', target_dir, records=[record_name], keep_subdirs=False)
            
        print("✅ SUCCESS: All biomedical data downloaded successfully.")

    except Exception as e:
        print(f"❌ ERROR: PHYSIONET Connection Failed: {e}")

if __name__ == "__main__":
    download_dataset()

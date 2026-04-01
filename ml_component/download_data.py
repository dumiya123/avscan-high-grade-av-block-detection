import os
import sys
import wfdb
from tqdm.auto import tqdm

import time

def main():
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'raw', 'ludb')
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Suppress verbose wfdb console output
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    print("Connecting to PhysioNet Database...")
    try:
        with HiddenPrints():
            records = wfdb.get_record_list('ludb')
    except Exception as e:
        print(f"❌ Failed to reach PhysioNet to get record list: {e}")
        return
        
    # Filter out files we already downloaded successfully
    missing_records = []
    for rec in records:
        if not (os.path.exists(os.path.join(DATA_DIR, f"{rec}.hea")) and os.path.exists(os.path.join(DATA_DIR, f"{rec}.dat"))):
            missing_records.append(rec)

    if not missing_records:
        print("✅ LUDB dataset already exists locally and is completely downloaded.")
        return

    print(f"Downloading {len(missing_records)} missing LUDB records. This will take a moment...\n")
    
    for rec in tqdm(missing_records, desc="Downloading LUDB ECGs"):
        retries = 3
        while retries > 0:
            try:
                with HiddenPrints():
                    wfdb.dl_database('ludb', dl_dir=DATA_DIR, records=[rec])
                break # Success! Break out of retry loop
            except Exception as e:
                retries -= 1
                if retries == 0:
                    print(f"\n❌ PhysioNet Server completely rejected {rec}: {str(e)}")
                    print("Please rerun the script later. The server is likely down.")
                    sys.exit(1)
                time.sleep(2) # Wait 2 seconds before hammering the server again
                
    print("\n✅ Dataset Download Complete!")

if __name__ == '__main__':
    main()

"""
Local LUDB Downloader.
Downloads the Lobachevsky University Database (LUDB) from PhysioNet.
"""
import os
import requests
from tqdm import tqdm
from pathlib import Path

BASE_URL = "https://physionet.org/files/ludb/1.0.1/"
DATA_DIR = Path("data/raw/ludb")

def download_file(url, dest, pbar=None):
    if dest.exists() and dest.stat().st_size > 0:
        if pbar: pbar.update(1)
        return
    os.makedirs(dest.parent, exist_ok=True)
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(dest, 'wb') as f:
                f.write(response.content)
        if pbar: pbar.update(1)
    except Exception as e:
        if pbar: pbar.update(1) # Still update to keep bar moving

def main():
    print(f"🚀 Initializing LUDB Synchronization...")
    
    # 1. Download metadata and labels list
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    
    download_file(f"{BASE_URL}RECORDS", DATA_DIR / "RECORDS")
    
    # 2. Get the list of all records
    with open(DATA_DIR / "RECORDS", 'r') as f:
        records = [line.strip() for line in f if line.strip()]

    # 3. Define extensions (Waveform, Header, Lead II annotations)
    # Note: Lead II (.ii) is the most critical for P-wave research
    extensions = [".dat", ".hea", ".ii"]
    
    # 4. Create full list of download tasks
    tasks = []
    for r in records:
        for ext in extensions:
            tasks.append((f"{BASE_URL}{r}{ext}", DATA_DIR / f"{r}{ext}"))

    print(f"📊 Total files to synchronize: {len(tasks)}")
    
    # 5. Download with OVERALL progress bar
    with tqdm(total=len(tasks), desc="Overall Progress", unit="file", colour='green') as pbar:
        for url, dest in tasks:
            download_file(url, dest, pbar)

    print("\n✅ LUDB Synchronization complete!")

if __name__ == "__main__":
    main()

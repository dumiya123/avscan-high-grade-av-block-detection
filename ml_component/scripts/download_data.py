"""
Dataset download module for LUDB and PTB-XL
"""

import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib
from typing import Optional


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        output_path: Path to save file
        chunk_size: Download chunk size
        
    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)
        
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def download_ludb(output_dir: str = "data/raw") -> bool:
    """
    Download LUDB dataset from PhysioNet using wget
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        True if successful
    """
    output_path = Path(output_dir) / "ludb"
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://physionet.org/files/ludb/1.0.1/"
    
    print(f"📥 Downloading LUDB dataset to {output_path}")
    print(f"   Source: {base_url}")
    
    try:
        # Use wget for recursive download
        cmd = [
            "wget",
            "-r",              # Recursive
            "-N",              # Only download newer files
            "-c",              # Continue partial downloads
            "-np",             # No parent directories
            "-nH",             # No host directories
            "--cut-dirs=3",    # Remove path components
            "-P", str(output_path),
            base_url
        ]
        
        # Try wget command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ LUDB download complete!")
            return True
        else:
            # Fallback to manual download of essential files
            print("⚠️  wget not available, using fallback method...")
            return download_ludb_fallback(output_path, base_url)
            
    except FileNotFoundError:
        # wget not installed, use fallback
        print("⚠️  wget not found, using fallback download method...")
        return download_ludb_fallback(output_path, base_url)


def download_ludb_fallback(output_path: Path, base_url: str) -> bool:
    """
    Fallback method to download LUDB files individually
    
    Args:
        output_path: Directory to save files
        base_url: Base URL
        
    Returns:
        True if successful
    """
    # Download essential files
    essential_files = [
        "RECORDS",
        "SHA256SUMS.txt"
    ]
    
    # Download record list first
    records_url = base_url + "RECORDS"
    records_path = output_path / "RECORDS"
    
    if download_file(records_url, records_path):
        # Read record names
        with open(records_path, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
        
        # Download each record (.dat, .hea, .atr files)
        for record in tqdm(records, desc="Downloading LUDB records"):
            for ext in ['.dat', '.hea', '.i']:
                file_url = base_url + record + ext
                file_path = output_path / (record + ext)
                download_file(file_url, file_path)
        
        print("✅ LUDB download complete!")
        return True
    else:
        print("❌ Failed to download LUDB")
        return False


def download_ptbxl(output_dir: str = "data/raw") -> bool:
    """
    Download PTB-XL dataset from PhysioNet
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        True if successful
    """
    output_path = Path(output_dir) / "ptbxl"
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    
    print(f"📥 Downloading PTB-XL dataset to {output_path}")
    print(f"   Source: {base_url}")
    print(f"   Note: This is a large dataset (~8GB), download may take time...")
    
    try:
        # Use wget for recursive download
        cmd = [
            "wget",
            "-r",
            "-N",
            "-c",
            "-np",
            "-nH",
            "--cut-dirs=3",
            "-P", str(output_path),
            base_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PTB-XL download complete!")
            return True
        else:
            print("⚠️  wget not available, using fallback method...")
            return download_ptbxl_fallback(output_path, base_url)
            
    except FileNotFoundError:
        print("⚠️  wget not found, using fallback download method...")
        return download_ptbxl_fallback(output_path, base_url)


def download_ptbxl_fallback(output_path: Path, base_url: str) -> bool:
    """
    Fallback method to download PTB-XL essential files
    
    Args:
        output_path: Directory to save files
        base_url: Base URL
        
    Returns:
        True if successful
    """
    # Download metadata files
    metadata_files = [
        "ptbxl_database.csv",
        "scp_statements.csv",
        "RECORDS",
        "RECORDS-waveforms"
    ]
    
    print("📥 Downloading PTB-XL metadata...")
    for filename in metadata_files:
        file_url = base_url + filename
        file_path = output_path / filename
        download_file(file_url, file_path)
    
    print("✅ PTB-XL metadata downloaded!")
    print("⚠️  Note: Full waveform data requires wget or manual download")
    print(f"   You can manually download from: {base_url}")
    
    return True


def verify_dataset(dataset_name: str, data_dir: str = "data/raw") -> bool:
    """
    Verify dataset integrity
    
    Args:
        dataset_name: 'ludb' or 'ptbxl'
        data_dir: Data directory
        
    Returns:
        True if dataset is valid
    """
    data_path = Path(data_dir) / dataset_name
    
    if not data_path.exists():
        print(f"❌ Dataset directory not found: {data_path}")
        return False
    
    if dataset_name == 'ludb':
        # Check for RECORDS file
        records_file = data_path / "RECORDS"
        if not records_file.exists():
            print(f"❌ RECORDS file not found in {data_path}")
            return False
        
        # Count .dat files
        dat_files = list(data_path.glob("*.dat"))
        print(f"✅ Found {len(dat_files)} LUDB records")
        return len(dat_files) > 0
    
    elif dataset_name == 'ptbxl':
        # Check for metadata
        metadata_file = data_path / "ptbxl_database.csv"
        if not metadata_file.exists():
            print(f"❌ Metadata file not found in {data_path}")
            return False
        
        print(f"✅ PTB-XL metadata found")
        return True
    
    return False


if __name__ == "__main__":
    # Test downloads
    print("Testing dataset downloads...")
    download_ludb()
    download_ptbxl()

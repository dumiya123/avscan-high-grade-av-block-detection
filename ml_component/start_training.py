"""
Simple training starter script - Run this to begin training
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed!")
        print(f"Error: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     AV Block Detection System - Training Starter         ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check if data exists
    processed_data = Path("data/processed/ecg_data.h5")
    
    if not processed_data.exists():
        print("⚠️  Processed data not found. Running full pipeline...\n")
        
        # Download data
        if not run_command(
            "python run.py download --datasets ludb",
            "Step 1: Downloading LUDB dataset"
        ):
            print("\n❌ Setup failed at download step")
            return
        
        # Preprocess data
        if not run_command(
            "python run.py preprocess --validate",
            "Step 2: Preprocessing data"
        ):
            print("\n❌ Setup failed at preprocessing step")
            return
    else:
        print("✅ Processed data found, skipping download and preprocessing\n")
    
    # Step 3: Start training
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    print("""
Training Configuration:
  - Epochs: 50
  - Batch Size: 16
  - Learning Rate: 0.0001
  - Segmentation Weight: 0.6
  - Classification Weight: 0.4
  - Early Stopping: Enabled (patience=15)

This will take approximately:
  - With GPU (RTX 3060): ~50 minutes
  - With CPU: ~8 hours

Press Ctrl+C to stop training at any time.
Checkpoints are saved after each epoch.
    """)
    
    input("Press Enter to start training...")
    
    # Train model
    run_command(
        "python run.py train --epochs 50 --batch-size 16",
        "Training Model"
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("""
Next steps:
  1. Evaluate the model:
     python run.py evaluate --checkpoint checkpoints/best_model.pth
  
  2. Run inference:
     python run.py inference --checkpoint checkpoints/best_model.pth --input sample.npy
  
  3. View training logs:
     tensorboard --logdir=logs
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nFor detailed troubleshooting, see TRAINING_GUIDE.md")


import numpy as np
import sys
import os
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).parent.parent / "ml_component"))

from src.inference.predictor import AVBlockPredictor
import json

def test_xai_output():
    # 1. Initialize predictor
    predictor = AVBlockPredictor()
    
    # 2. Create dummy signal
    # 5000 samples (10s at 500Hz)
    dummy_signal = np.sin(np.linspace(0, 50, 5000))
    
    # 3. Running prediction
    result = predictor.predict(dummy_signal)
    
    # 4. Verify XAI keys
    xai = result.get('xai', {})
    print(f"Diagnosis: {result['diagnosis']['av_block_type']}")
    print(f"XAI Keys: {list(xai.keys())}")
    print(f"Focus Label: {xai.get('focus_label')}")
    
    # Check if heatmap is present and has correct length
    heatmap = xai.get('heatmap', [])
    print(f"Heatmap Length: {len(heatmap)}")
    
    if len(heatmap) == 5000 and 'focus_label' in xai:
        print("\nSUCCESS: XAI structured output is correct.")
    else:
        print("\nFAILURE: XAI output is missing keys or has wrong dimensions.")

if __name__ == "__main__":
    test_xai_output()

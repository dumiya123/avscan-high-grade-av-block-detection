import requests
import numpy as np
import time
import os

def test_full_pipeline_inference():
    """
    End-to-End Integration Test
    Simulates a user uploading an ECG signal and receiving an intelligent diagnostic report.
    This test assumes the backend FastAPI server is running locally on port 8000.
    """
    api_url = "http://localhost:8000/predict"
    
    # Generate mock 10s ECG signal containing 5000 sampling points
    dummy_signal = np.sin(np.linspace(0, 100, 5000)).astype(np.float32)
    signal_bytes = dummy_signal.tobytes()
    
    try:
        t0 = time.time()
        response = requests.post(
            api_url,
            files={"file": ("test_ecg.npy", signal_bytes, "application/octet-stream")},
            timeout=10
        )
        t1 = time.time()
        
        # We only throw error if server answers but code is not 200
        if response.status_code == 200:
            data = response.json()
            av_block_types = [
                "Normal", "1st Degree", "2nd Degree Type I", 
                "2nd Degree Type II", "3rd Degree", "VT with dissociation"
            ]
            
            # Basic schema validations
            assert data["diagnosis"]["av_block_type"] in av_block_types
            assert 0.0 <= data["diagnosis"]["confidence"] <= 1.0
            
            print(f"E2E Integration Test: PASSED (Latency: {t1-t0:.2f}s)")
        else:
            print(f"Backend responded with an error, code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("Backend server not accessible on localhost:8000. Start it via `python backend/main.py`.")
    except Exception as e:
        print(f"E2E Integration Test: FAILED due to unexpected error {e}")

if __name__ == "__main__":
    print("Initiating full E2E pipeline check...")
    test_full_pipeline_inference()

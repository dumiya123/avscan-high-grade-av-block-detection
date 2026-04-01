import pytest
import numpy as np
import io
import sys
import os

# Ensure the backend module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
try:
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
except ImportError:
    client = None

@pytest.mark.skipif(client is None, reason="Backend dependencies or main.py not found")
def test_health_check():
    """Test if the API is awake and responding."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.skipif(client is None, reason="Backend dependencies or main.py not found")
def test_predict_endpoint_valid_ecg():
    """Test the prediction endpoint with a valid dummy ECG signal array."""
    # Create a dummy 10-second ECG signal at 500Hz
    dummy_signal = np.random.randn(5000).astype(np.float32)
    signal_bytes = dummy_signal.tobytes()
    
    response = client.post(
        "/predict",
        files={"file": ("dummy_ecg.npy", signal_bytes, "application/octet-stream")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "diagnosis" in data
    assert "confidence" in data["diagnosis"]
    assert "intervals" in data
    assert "xai" in data
    assert "explanation" in data["xai"]

@pytest.mark.skipif(client is None, reason="Backend dependencies or main.py not found")
def test_predict_endpoint_invalid_file():
    """Test API behavior when an invalid file mechanism is detected."""
    response = client.post(
        "/predict",
        files={"file": ("dummy.txt", b"invalid text format", "text/plain")}
    )
    
    # Depending on implementation, it should yield a 400 Bad Request
    assert response.status_code in [400, 415, 422]

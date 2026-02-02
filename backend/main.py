from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
import torch
from pathlib import Path
import sys
import os
import shutil
import base64
from typing import Dict

# Add ml_component directory to path to import src
sys.path.append(str(Path(__file__).parent.parent / "ml_component"))

from src.inference.predictor import AVBlockPredictor

app = FastAPI(title="AtrionNet Clinical API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CHECKPOINT_PATH = Path(__file__).parent.parent / "ml_component/checkpoints/best_model.pth"
UPLOAD_DIR = Path(__file__).parent / "uploads"
REPORT_DIR = Path(__file__).parent / "reports"

# Initialize predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from {CHECKPOINT_PATH}...")
        predictor = AVBlockPredictor(checkpoint=CHECKPOINT_PATH)
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. API running in restricted mode.")

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "model_loaded": predictor is not None,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    }

@app.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    
    if not file.filename.endswith('.npy'):
        raise HTTPException(status_code=400, detail="Only .npy files are supported")
    
    try:
        # Save temporary file
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load and predict
        signal = np.load(temp_path)
        result = predictor.predict(signal)
        
        # Generate report path
        report_id = f"report_{Path(file.filename).stem}.pdf"
        report_path = REPORT_DIR / report_id
        
        # Save report
        predictor.save_report(result, report_path)
        
        # Clean up input file
        os.remove(temp_path)
        
        # Prepare response (convert numpy arrays to lists for JSON)
        # Subsample signal to 1000 points for smooth frontend plotting if long
        viz_signal = signal.flatten()
        if len(viz_signal) > 1000:
            indices = np.linspace(0, len(viz_signal) - 1, 1000).astype(int)
            viz_signal = viz_signal[indices]

        response_data = {
            "diagnosis": result['diagnosis']['av_block_type'],
            "confidence": float(result['diagnosis']['confidence']),
            "severity": result['diagnosis']['severity'],
            "intervals": {
                **result['intervals'],
                "avg_pr": float(np.mean(result['intervals']['pr'])) if result['intervals']['pr'] else 0,
                "hr": 60.0 / (np.mean(result['intervals']['rr']) / 500.0) if result['intervals']['rr'] else 72.0
            },
            "report_id": report_id,
            "explanation": result['xai']['explanation'],
            "signal": viz_signal.tolist()
        }
        
        return response_data
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    report_path = REPORT_DIR / report_id
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(report_path, media_type="application/pdf", filename=report_id)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

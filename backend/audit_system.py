import os
import sys
import torch
import numpy as np
from pathlib import Path

def audit():
    print("="*60)
    print("      ATRIONNET SYSTEM RELIABILITY AUDIT")
    print("="*60)
    
    # 1. Environment Check
    print(f"\n[1/4] Environment Audit:")
    print(f" - Python Version : {sys.version}")
    print(f" - Torch Version  : {torch.__version__}")
    print(f" - CUDA Available : {torch.cuda.is_available()}")
    
    # 2. Path Resolution
    print(f"\n[2/4] File System Audit:")
    backend_root = Path(__file__).parent
    project_root = backend_root.parent
    checkpoint_path = project_root / "ml_component/outputs/weights/atrion_hybrid_best.pth"
    
    print(f" - Backend Path   : {backend_root}")
    print(f" - Project Root   : {project_root}")
    print(f" - Target Weights : {checkpoint_path}")
    
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f" - Weights Found  : YES ({size_mb:.2f} MB)")
    else:
        print(f" - Weights Found  : [FAIL] NO (Path mismatch or file missing)")
        return

    # 3. Predictor Integration Audit
    print(f"\n[3/4] Model Integration Audit:")
    sys.path.append(str(project_root / "ml_component"))
    
    try:
        from src.inference.predictor import AVBlockPredictor
        print(" - Predictor Import: OK")
        
        # Check if the predictor successfully initializes in 'model' mode
        predictor = AVBlockPredictor(checkpoint=checkpoint_path)
        print(f" - Internal Mode  : {predictor._mode}")
        
        if predictor._mode == "fallback":
            print("   [WARN] Predictor is in Fallback mode despite weights being present.")
        else:
            print("   [SUCCESS] Predictor is in AI-Engine Mode.")
            
        # 4. Inference Logic Audit
        print(f"\n[4/4] Clinical Ground-Truth Audit:")
        test_file = project_root / "test_case_34.npy"
        if test_file.exists():
            print(f" - Loading Gold Case: test_case_34.npy")
            test_signal = np.load(test_file)
            
            # Internal model check
            with torch.no_grad():
                x = torch.tensor(test_signal[np.newaxis]).to(predictor.device).float()
                out = predictor.model(x)
                h = out['heatmap'][0].cpu().numpy()
                print(f" - Raw Heatmap Max : {h.max():.4f}")
                print(f" - Raw Heatmap Min : {h.min():.4f}")
                print(f" - Raw Heatmap Mean: {h.mean():.4f}")

            result = predictor.predict(test_signal)
            
            conf = result['diagnosis']['confidence']
            diag = result['diagnosis']['av_block_type']
            severity = result['diagnosis']['severity']
            
            print(f" - Model Diagnosis : {diag}")
            print(f" - Clinical Severity: {severity}")
            print(f" - AI Confidence   : {conf * 100:.1f}%")
            
            # Record 104 is a 3rd Degree Block per LUDB .hea
            if "3rd" in diag or "Complete" in diag or "Dissociation" in diag:
                print("   [SUCCESS] The AI correctly identified the 3rd Degree AV block.")
            else:
                print("   [FAILED] The AI missed the block or misclassified it.")
        else:
            print(" - Gold Case Missing: Skipping Clinical Audit")

    except Exception as e:
        print(f" - Audit Failed   : [FAIL] {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("      AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    audit()

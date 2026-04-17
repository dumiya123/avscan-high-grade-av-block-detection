import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def compute_segmentation_metrics(tp, fp, fn, mAP=None):
    """
    Computes Precision, Recall (Sensitivity in AAMI EC57), F1, and includes mAP.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "mAP @ 0.5": mAP if mAP is not None else 0.0
    }

def compute_clinical_detection_metrics(y_true, y_pred, y_scores=None):
    """
    Computes overall clinical detection metrics.
    y_true: True binary labels (e.g., patient has dissociated P-waves = 1)
    y_pred: Predicted binary labels
    y_scores: Predicted probabilities (for AUC)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }
    
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            metrics["ROC_AUC"] = auc
        except ValueError:
            metrics["ROC_AUC"] = None
            
    return metrics

def calculate_computational_performance(model, input_size=(1, 12, 5000), device='cpu'):
    """
    Calculates model parameters and estimates inference time.
    """
    import torch
    import time
    
    model.to(device)
    model.eval()
    
    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Inference time (batch_size = 1)
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end = time.time()
    
    avg_inference_time_ms = ((end - start) / 100) * 1000
    
    return {
        "Total_Parameters": total_params,
        "Trainable_Parameters": trainable_params,
        "Inference_Time_ms": avg_inference_time_ms
    }

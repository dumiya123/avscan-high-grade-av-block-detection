"""
AtrionNet Research Metrics: 1D IoU and mean Average Precision (mAP).
This module validates the 'Instance-Level' detection performance of our anchor-free model.
"""

import numpy as np
import torch
from scipy.signal import find_peaks

def calculate_1d_iou(pred_mask, target_mask):
    """
    Computes Intersection over Union for 1D sample masks.
    """
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 0
    return intersection / union

def get_instances_from_heatmap(heatmap, width_map, threshold=0.5):
    """
    Extracts individual wave instances from the model's heatmap and width heads.
    This is the core of our 'Instance-Isolation' logic.
    """
    heatmap = heatmap.squeeze()
    width_map = width_map.squeeze()
    
    # 1. Find local maxima in the heatmap (Centers of P-waves)
    peaks, _ = find_peaks(heatmap, height=threshold, distance=20)
    
    instances = []
    for peak in peaks:
        center = peak
        w = width_map[peak] * 5000 # Scaling back to sample length
        
        start = max(0, int(center - w/2))
        end = min(len(heatmap), int(center + w/2))
        
        instances.append({
            'center': center,
            'width': w,
            'span': (start, end),
            'confidence': heatmap[peak]
        })
    return instances

def compute_instance_metrics(pred_heatmap, pred_width, target_instances, iou_threshold=0.5):
    """
    Calculates 1D Average Precision for a single ECG.
    """
    preds = get_instances_from_heatmap(pred_heatmap, pred_width)
    
    if len(preds) == 0:
        return 0.0 if len(target_instances) > 0 else 1.0

    # Sort by confidence
    preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
    
    tp = 0
    fp = 0
    matched_targets = set()

    for pred in preds:
        best_iou = 0
        best_target_idx = -1
        
        p_start, p_end = pred['span']
        pred_mask = np.zeros(5000)
        pred_mask[p_start:p_end] = 1
        
        for idx, target in enumerate(target_instances):
            if idx in matched_targets:
                continue
            
            t_start, t_end = target['span']
            target_mask = np.zeros(5000)
            target_mask[t_start:t_end] = 1
            
            iou = calculate_1d_iou(pred_mask, target_mask)
            if iou > best_iou:
                best_iou = iou
                best_target_idx = idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_targets.add(best_target_idx)
        else:
            fp += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / len(target_instances) if len(target_instances) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'tp': tp,
        'fp': fp,
        'fn': len(target_instances) - tp
    }

if __name__ == "__main__":
    # Example logic verification
    h = np.zeros(5000)
    h[1000] = 0.9 # Center 1
    h[2000] = 0.8 # Center 2
    w = np.zeros(5000)
    w[1000] = 0.04 # 200 samples wide
    w[2000] = 0.03 # 150 samples wide
    
    instances = get_instances_from_heatmap(h, w)
    print(f"Detected {len(instances)} P-wave instances.")
    for i, inst in enumerate(instances):
        print(f"  Instance {i+1}: Center={inst['center']}, Span={inst['span']}")

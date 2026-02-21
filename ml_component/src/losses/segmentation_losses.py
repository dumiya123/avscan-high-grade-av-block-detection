"""
AtrionNet Research Losses.
Implements Focal Loss, Dice Loss, and the Multi-Task Instance Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, target, alpha=2.5, beta=4.0):
    """
    RESEARCH-TUNED FOCAL LOSS.
    Modified to prioritize 'Precision' over 'Recall'.
    alpha=2.5: Increases penalty for misclassifying P-waves.
    beta=4.0: Aggressively suppresses False Positives (T-waves/Noise).
    """
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    # We use (1 - target) to handle the 'blurred' Gaussian edges
    neg_weights = torch.pow(1 - target, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum() # Handle cases with no P-waves
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos

def create_instance_loss(pred, target):
    """
    STRICT RESEARCH LOSS.
    Weighted to solve the Precision-Recall imbalance found in real LUDB data.
    """
    import torch.nn.functional as F
    
    # 1. Detection (Increased penalty for False Alarms)
    hm_loss = focal_loss(pred['heatmap'], target['heatmap'])
    
    # 2. Width Quantification
    center_mask = target['heatmap'] > 0.98 # Tighter center focus
    if center_mask.sum() > 0:
        w_loss = F.smooth_l1_loss(pred['width'][center_mask], target['width'][center_mask])
    else:
        w_loss = torch.tensor(0.0).to(pred['width'].device)
        
    # 3. Mask Segmentation
    m_loss = F.binary_cross_entropy(pred['mask'], target['mask']) + dice_loss(pred['mask'], target['mask'])
    
    # Final Research Balancing:
    # 30x weighting on Heatmap to force the model to stop 'guessing'
    return (30.0 * hm_loss) + (1.0 * w_loss) + (5.0 * m_loss)

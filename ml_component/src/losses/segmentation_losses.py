"""
AtrionNet Research Losses.
Implements Focal Loss, Dice Loss, and the Multi-Task Instance Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, target, alpha=2, beta=4):
    """Refined Focal Loss for Heatmaps."""
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for mask segmentation."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def create_instance_loss(pred, target):
    """RESEARCH-GRADE MULTI-TASK LOSS."""
    # 1. Heatmap (Heavy weight for detection)
    hm_loss = focal_loss(pred['heatmap'], target['heatmap'])
    
    # 2. Width (SmoothL1 for quantification stability)
    center_mask = target['heatmap'] > 0.95
    if center_mask.sum() > 0:
        w_loss = F.smooth_l1_loss(pred['width'][center_mask], target['width'][center_mask])
    else:
        w_loss = torch.tensor(0.0).to(pred['width'].device)
        
    # 3. Mask (BCE + Dice for robust segmentation)
    m_loss = F.binary_cross_entropy(pred['mask'], target['mask']) + dice_loss(pred['mask'], target['mask'])
    
    # Final Balance
    return (15.0 * hm_loss) + (1.0 * w_loss) + (5.0 * m_loss)

"""
AtrionNet Research Losses.
Implements Focal Loss, Dice Loss, and the Multi-Task Instance Loss.
"""
import torch
import torch.nn.functional as F

def focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    RESEARCH-TUNED FOCAL LOSS.
    To prevent low confidence scores (which cause FN drops when thresholds are raised),
    we enforce a stronger penalty on missing the exact center of P-waves.
    """
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    # We use (1 - target) to handle the 'blurred' Gaussian edges
    neg_weights = torch.pow(1 - target, beta)

    # Remove the multiplier, it destroys the natural gradient scaling.
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum() 
    return -(pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-6)


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for mask segmentation."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def create_instance_loss(pred, target):
    """
    CLEAN RESEARCH LOSS.
    Balanced ratios without extreme overriding values.
    """
    # 1. Detection
    hm_loss = focal_loss(pred['heatmap'], target['heatmap'])
    
    # 2. Width Quantification
    center_mask = target['heatmap'] > 0.999
    if center_mask.sum() > 0:
        w_loss = F.smooth_l1_loss(pred['width'][center_mask], target['width'][center_mask])
    else:
        w_loss = torch.tensor(0.0).to(pred['width'].device)
        
    # 3. Mask Segmentation
    m_loss = F.binary_cross_entropy(pred['mask'], target['mask']) + dice_loss(pred['mask'], target['mask'])
    
    # Final Research Balancing:
    return (10.0 * hm_loss) + (1.0 * w_loss) + (2.0 * m_loss)

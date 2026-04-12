"""
AtrionNet Research Losses.
Implements Focal Loss, Dice Loss, and the Multi-Task Instance Loss.
"""
import torch
import torch.nn.functional as F

def focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    RESEARCH-TUNED FOCAL LOSS.
    Modified to consider pixels with target >= 0.8 as positive to broaden the gradient surface.
    """
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_inds = target.ge(0.8).float()
    neg_inds = target.lt(0.8).float()
    
    # We use (1 - target) to handle the 'blurred' Gaussian edges
    neg_weights = torch.pow(1 - target, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.mean() 
    return -(pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-6)


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for mask segmentation."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def create_instance_loss(pred, target, alpha=2.0, beta=4.0, hm_weight=2.0, w_weight=1.0, m_weight=1.0):
    """
    CLEAN RESEARCH LOSS.
    Balanced ratios without extreme overriding values.
    """
    # 1. Detection
    hm_loss = focal_loss(pred['heatmap'], target['heatmap'], alpha=alpha, beta=beta)
    
    # 2. Width Quantification (Using broader mask target)
    center_mask = target['mask'] > 0.5
    if center_mask.sum() > 0:
        w_loss = F.smooth_l1_loss(pred['width'][center_mask], target['width'][center_mask])
    else:
        # Dummy loss with requires_grad=True to prevent disconnected graph
        w_loss = (pred['width'] * 0.0).sum()
        
    # 3. Mask Segmentation — Weighted BCE to counter 80/20 class imbalance
    pos_weight = torch.tensor([5.0], device=pred['mask'].device)
    m_bce = F.binary_cross_entropy_with_logits(
        pred['mask'], target['mask'],
        pos_weight=pos_weight
    )
    m_loss = m_bce + dice_loss(torch.sigmoid(pred['mask']), target['mask'])
    
    # Final Research Balancing (Ablated):
    return (hm_weight * hm_loss) + (w_weight * w_loss) + (m_weight * m_loss)


"""
Loss Functions for Multi-Task Learning (Segmentation + Classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights well-classified examples.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation overlap optimization.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, L) logits
        # targets: (B, L) class indices
        
        num_classes = inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 2, 1).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()

class MultiTaskLoss(nn.Module):
    """
    Combined Loss for:
    1. Segmentation (Focal + Dice)
    2. Classification (Cross Entropy)
    """
    def __init__(self, seg_weight=0.6, clf_weight=0.4):
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.clf_weight = clf_weight
        
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_target, clf_pred, clf_target):
        # Segmentation Loss
        focal = self.focal(seg_pred, seg_target)
        dice = self.dice(seg_pred, seg_target)
        seg_loss = focal + dice
        
        # Classification Loss
        clf_loss = self.ce(clf_pred, clf_target)
        
        # Weighted Sum
        total_loss = (self.seg_weight * seg_loss) + (self.clf_weight * clf_loss)
        
        return total_loss, seg_loss, clf_loss

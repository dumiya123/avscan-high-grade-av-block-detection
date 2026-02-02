"""
Custom loss functions for ECG segmentation and classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (list or tensor)
            gamma: Focusing parameter
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes, seq_len)
            targets: (batch, seq_len)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes, seq_len)
            targets: (batch, seq_len)
        """
        num_classes = inputs.shape[1]
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 2, 1).float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Calculate Dice coefficient
        intersection = (inputs_soft * targets_one_hot).sum(dim=2)
        union = inputs_soft.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss
    """
    
    def __init__(self, alpha=None, gamma=2.0, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for segmentation + classification
    """
    
    def __init__(self, seg_weight=0.6, clf_weight=0.4, 
                 seg_class_weights=None, clf_class_weights=None):
        """
        Args:
            seg_weight: Weight for segmentation loss
            clf_weight: Weight for classification loss
            seg_class_weights: Class weights for segmentation
            clf_class_weights: Class weights for classification
        """
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.clf_weight = clf_weight
        
        # Segmentation loss (Focal + Dice)
        self.seg_loss = CombinedLoss(alpha=seg_class_weights, gamma=2.0, dice_weight=0.5)
        
        # Classification loss (CrossEntropy)
        self.clf_loss = nn.CrossEntropyLoss(weight=clf_class_weights)
    
    def forward(self, seg_pred, seg_target, clf_pred, clf_target):
        """
        Args:
            seg_pred: Segmentation predictions (batch, num_classes, seq_len)
            seg_target: Segmentation targets (batch, seq_len)
            clf_pred: Classification predictions (batch, num_classes)
            clf_target: Classification targets (batch,)
            
        Returns:
            total_loss, seg_loss, clf_loss
        """
        seg_loss = self.seg_loss(seg_pred, seg_target)
        clf_loss = self.clf_loss(clf_pred, clf_target)
        
        total_loss = self.seg_weight * seg_loss + self.clf_weight * clf_loss
        
        return total_loss, seg_loss, clf_loss


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    num_seg_classes = 5
    num_clf_classes = 6
    seq_len = 1000
    
    # Dummy data
    seg_pred = torch.randn(batch_size, num_seg_classes, seq_len)
    seg_target = torch.randint(0, num_seg_classes, (batch_size, seq_len))
    clf_pred = torch.randn(batch_size, num_clf_classes)
    clf_target = torch.randint(0, num_clf_classes, (batch_size,))
    
    # Test multi-task loss
    loss_fn = MultiTaskLoss(seg_weight=0.6, clf_weight=0.4)
    total_loss, seg_loss, clf_loss = loss_fn(seg_pred, seg_target, clf_pred, clf_target)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Segmentation loss: {seg_loss.item():.4f}")
    print(f"Classification loss: {clf_loss.item():.4f}")

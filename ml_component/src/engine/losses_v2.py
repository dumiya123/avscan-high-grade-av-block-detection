"""
Loss Functions for Multi-Label ECG Classification.

Supports:
- BCEWithLogitsLoss (with optional class weights)
- Focal Loss (for severe class imbalance)

Both operate on raw logits (no sigmoid needed before calling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Standard BCEWithLogitsLoss with per-class weighting.
    Numerically stable: operates on raw logits internally.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        """
        Args:
            pos_weight: (num_classes,) tensor of positive class weights.
                        Higher weight = more penalty for missing that class.
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Args:
            logits:  (batch, num_classes) raw model output
            targets: (batch, num_classes) multi-hot labels
        """
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class MultilabelFocalLoss(nn.Module):
    """
    Focal Loss adapted for multi-label classification.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Addresses class imbalance by down-weighting easy examples
    and focusing training on hard, misclassified ones.

    Reference: Lin et al. (2017) - Focal Loss for Dense Object Detection.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, pos_weight: torch.Tensor = None):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   gamma=0 reduces to standard BCE.
            alpha: Balancing factor for positive vs negative.
            pos_weight: Optional per-class weights.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Args:
            logits:  (batch, num_classes) raw logits
            targets: (batch, num_classes) multi-hot labels {0, 1}
        """
        # Standard BCE (per element, no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Probability of correct class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Per-class weighting (optional)
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            class_weight = pw * targets + (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss * class_weight
        else:
            focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()


def get_loss_function(loss_type: str = 'bce', pos_weight: torch.Tensor = None, **kwargs):
    """
    Factory for loss functions.

    Args:
        loss_type: 'bce' or 'focal'
        pos_weight: Class weights tensor.
        **kwargs: Additional args (gamma, alpha for focal).

    Returns:
        nn.Module loss function.
    """
    if loss_type == 'bce':
        return WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == 'focal':
        return MultilabelFocalLoss(
            gamma=kwargs.get('gamma', 2.0),
            alpha=kwargs.get('alpha', 0.25),
            pos_weight=pos_weight,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'bce' or 'focal'.")

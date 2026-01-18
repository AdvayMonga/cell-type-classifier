"""Focal Loss implementation for handling class imbalance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Focuses learning on hard-to-classify examples by down-weighting
    easy examples. Useful for imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes] or None for equal weights)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(labels, method='inverse'):
    """
    Compute class weights from label distribution.

    Args:
        labels: Array of class labels
        method: 'inverse' (1/freq), 'sqrt_inverse' (1/sqrt(freq)), or 'balanced'

    Returns:
        torch.Tensor of class weights
    """
    import numpy as np

    unique, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)

    if method == 'inverse':
        weights = n_samples / (n_classes * counts)
    elif method == 'sqrt_inverse':
        weights = np.sqrt(n_samples / (n_classes * counts))
    elif method == 'balanced':
        weights = n_samples / counts
        weights = weights / weights.sum() * n_classes
    else:
        raise ValueError(f"Unknown method: {method}")

    weights = weights / weights.max()
    return torch.FloatTensor(weights)

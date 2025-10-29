"""
Advanced loss functions for improving model performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002

    Focuses training on hard examples by down-weighting easy examples
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for classes (tensor of size num_classes)
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
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss

    Prevents the model from becoming over-confident
    Helps with generalization and calibration
    """

    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing, 1 = uniform distribution)
            reduction: 'mean', 'sum', or 'none'
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size)

        Returns:
            Label smoothing loss value
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed labels
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        loss = (-targets_smooth * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions
    Useful for leveraging benefits of different losses
    """

    def __init__(self, losses, weights=None):
        """
        Args:
            losses: List of loss functions
            weights: List of weights for each loss (default: equal weights)
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleList(losses)

        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        self.weights = weights

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions
            targets: Ground truth labels

        Returns:
            Weighted combination of losses
        """
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)

        return total_loss


class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross Entropy Loss
    More robust to label noise

    Reference: https://arxiv.org/abs/1908.06112
    """

    def __init__(self, alpha=0.1, beta=1.0, num_classes=3):
        """
        Args:
            alpha: Weight for reverse cross entropy
            beta: Weight for standard cross entropy
            num_classes: Number of classes
        """
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size)

        Returns:
            Symmetric cross entropy loss
        """
        # Standard cross entropy
        ce = F.cross_entropy(inputs, targets)

        # Reverse cross entropy
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        rce = (-targets_one_hot * torch.log(pred)).sum(dim=1)
        rce = rce.mean()

        # Symmetric loss
        loss = self.alpha * rce + self.beta * ce

        return loss


def get_loss_function(loss_type='cross_entropy', **kwargs):
    """
    Factory function to get loss function by name

    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing', 'symmetric')
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function
    """
    if loss_type == 'cross_entropy':
        weight = kwargs.get('class_weights', None)
        return nn.CrossEntropyLoss(weight=weight)

    elif loss_type == 'focal':
        alpha = kwargs.get('class_weights', None)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    elif loss_type == 'symmetric':
        alpha = kwargs.get('alpha', 0.1)
        beta = kwargs.get('beta', 1.0)
        num_classes = kwargs.get('num_classes', 3)
        return SymmetricCrossEntropy(alpha=alpha, beta=beta, num_classes=num_classes)

    elif loss_type == 'combined':
        # Example: focal + label smoothing
        focal = FocalLoss(alpha=kwargs.get('class_weights', None), gamma=2.0)
        label_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
        return CombinedLoss([focal, label_smooth], weights=[0.7, 0.3])

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

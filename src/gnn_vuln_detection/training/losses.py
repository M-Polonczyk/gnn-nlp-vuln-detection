import torch
import torch.nn.functional as F


def cross_entropy_loss_with_weights(outputs, targets, weight_for_positive_class=1.0):
    """
    Computes weighted cross-entropy loss, useful for imbalanced datasets
    where the positive class (vulnerable) might be rare.

    Args:
        outputs (torch.Tensor): Model outputs (logits).
        targets (torch.Tensor): True labels.
        weight_for_positive_class (float): Weight to apply to the positive class.

    Returns:
        torch.Tensor: Weighted cross-entropy loss.
    """
    # Assuming targets are 0 for negative, 1 for positive
    weights = torch.ones_like(targets, dtype=torch.float)
    weights[targets == 1] = weight_for_positive_class

    return F.cross_entropy(outputs, targets, weight=weights)


def focal_loss(outputs, targets, alpha=1.0, gamma=2.0):
    """
    Computes Focal Loss, which is designed to address class imbalance by down-weighting easy examples
    and focusing training on hard negatives.

    Args:
        outputs (torch.Tensor): Model outputs (logits).
        targets (torch.Tensor): True labels.
        alpha (float): Weighting factor for the positive class.
        gamma (float): Focusing parameter.

    Returns:
        torch.Tensor: Focal loss.
    """
    BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(-BCE_loss)  # Probability of true class
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()

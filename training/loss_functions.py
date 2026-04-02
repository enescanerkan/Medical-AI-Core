# loss_functions.py
"""
Description: Custom loss functions optimized for medical image tasks (e.g., Dice Loss).
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Calculates the Dice Loss, which measures spatial overlap.

    IMPORTANT CONCEPT: Dice Loss is highly effective for imbalanced medical datasets
    (like finding a small tumor in a large MRI scan) where standard BCE loss fails.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): A small constant added to numerator and denominator
                            to avoid division by zero errors.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice Loss between predictions and ground truth.

        Args:
            predictions (torch.Tensor): The output probabilities from the model.
            targets (torch.Tensor): The binary ground truth masks.

        Returns:
            torch.Tensor: The calculated scalar loss value.
        """
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (predictions * targets).sum()

        dice_score = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice_score
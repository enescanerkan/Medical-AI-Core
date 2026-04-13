"""
Description: Computes domain-specific evaluation metrics for medical AI classification.
"""

import torch

class MedicalMetrics:
    """
    Provides static methods for evaluating medical models.
    """

    @staticmethod
    def calculate_sensitivity(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculates Sensitivity (Recall / True Positive Rate).
        Out of all actual positive cases, how many did we predict correctly?
        """
        preds_binary = (predictions > threshold).float()
        true_positives = (preds_binary * targets).sum()
        actual_positives = targets.sum()

        if actual_positives == 0:
            return 1.0

        return (true_positives / actual_positives).item()

    @staticmethod
    def calculate_specificity(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculates Specificity (True Negative Rate).
        Out of all actual negative cases, how many did we predict correctly?
        """
        preds_binary = (predictions > threshold).float()
        true_negatives = ((1 - preds_binary) * (1 - targets)).sum()
        actual_negatives = (1 - targets).sum()

        if actual_negatives == 0:
            return 1.0

        return (true_negatives / actual_negatives).item()

    @staticmethod
    def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculates overall Accuracy.
        """
        preds_binary = (predictions > threshold).float()
        correct = (preds_binary == targets).sum()
        total = targets.numel()

        return (correct / total).item()
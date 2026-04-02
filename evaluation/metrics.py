"""
Description: Computes domain-specific evaluation metrics for medical AI.
"""

import torch


class MedicalMetrics:
    """
    Provides static methods for evaluating medical models.

    IMPORTANT CONCEPT: In Medical AI, 'Accuracy' is a dangerous metric due to Class Imbalance.
    Sensitivity and IoU are preferred.
    """

    @staticmethod
    def calculate_iou(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculates Intersection over Union (Jaccard Index).

        IMPORTANT CONCEPT: IoU is the gold standard for Image Segmentation.

        Args:
            predictions (torch.Tensor): Model output probabilities.
            targets (torch.Tensor): Ground truth binary masks.
            threshold (float): Cutoff for binary classification.

        Returns:
            float: The IoU score between 0.0 and 1.0.
        """
        preds_binary = (predictions > threshold).float()
        intersection = (preds_binary * targets).sum()
        union = preds_binary.sum() + targets.sum() - intersection

        if union == 0:
            return 1.0

        return (intersection / union).item()

    @staticmethod
    def calculate_sensitivity(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculates Sensitivity (Recall).

        IMPORTANT CONCEPT: Sensitivity answers: "Out of all sick patients, how many did we find?".
        Maximizing this minimizes fatal False Negatives.

        Args:
            predictions (torch.Tensor): Model output probabilities.
            targets (torch.Tensor): Ground truth binary labels.
            threshold (float): Cutoff for binary classification.

        Returns:
            float: Sensitivity score between 0.0 and 1.0.
        """
        preds_binary = (predictions > threshold).float()
        true_positives = (preds_binary * targets).sum()
        actual_positives = targets.sum()

        if actual_positives == 0:
            return 1.0

        return (true_positives / actual_positives).item()
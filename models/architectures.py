# architectures.py
"""
Description: Defines Deep Learning architectures (e.g., CNNs, UNet) for medical image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMedicalCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for baseline medical image classification.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        """
        Initializes the network layers.

        Args:
            in_channels (int): Number of input channels (1 for grayscale medical scans).
            num_classes (int): Number of output classes (e.g., 2 for Benign/Malignant).
        """
        super(SimpleMedicalCNN, self).__init__()

        # Feature extraction block
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification head (assumes 224x224 input)
        self.fc1 = nn.Linear(16 * 112 * 112, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Raw class predictions (logits).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten spatial features
        x = self.fc1(x)
        return x
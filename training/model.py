import torch
import torch.nn as nn
from torchvision import models


class MammographyResNet(nn.Module):
    """
    A Convolutional Neural Network (CNN) based on the ResNet50 architecture,
    customized for binary classification of mammography images (e.g., MLO vs CC).

    Attributes:
        backbone (nn.Module): The pre-trained ResNet50 feature extractor.
    """

    def __init__(self, num_classes: int = 1):
        """
        Initializes the model by loading a pre-trained ResNet50 and replacing
        its final fully connected layer to match the required number of classes.

        Args:
            num_classes (int, optional): The number of output nodes. Defaults to 1
                                         for binary classification using BCEWithLogitsLoss.
        """
        super(MammographyResNet, self).__init__()

        # Load the pre-trained ResNet50 model using modern weights parameter
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Extract the number of input features for the final fully connected layer
        num_ftrs = self.backbone.fc.in_features

        # Replace the final classification layer
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): A batch of input images as a tensor.
                              Expected shape: (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: The raw, unnormalized predictions (logits) of the model.
        """
        return self.backbone(x)
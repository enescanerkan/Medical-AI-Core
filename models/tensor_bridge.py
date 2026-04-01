# tensor_bridge.py
"""
Acts as an interface between classical OpenCV arrays and PyTorch/TensorFlow tensors.
Ensures data is correctly typed, shaped, and normalized for Deep Learning models.
"""

import torch
import torchvision.transforms as T
import numpy as np


class TensorBridge:
    """
    Converts processed numpy arrays into model-ready tensors.
    Follows the Open/Closed Principle by allowing easy addition of new frameworks (e.g., TensorFlow) later.
    """

    def __init__(self, target_size: tuple = (224, 224)):
        """
        Initializes the transformation pipeline for PyTorch.

        Args:
            target_size (tuple): Desired image dimensions (H, W).
        """
        # Define standard transformations for medical grayscale images
        self.pytorch_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def to_pytorch(self, processed_image: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a normalized PyTorch Tensor.

        Args:
            processed_image (np.ndarray): The OpenCV image array.

        Returns:
            torch.Tensor: Ready-to-use tensor for model input.
        """
        # Ensure the image has a channel dimension (needed for grayscale in PyTorch)
        if len(processed_image.shape) == 2:
            processed_image = np.expand_dims(processed_image, axis=-1)

        return self.pytorch_transforms(processed_image)
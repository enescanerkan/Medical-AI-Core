# smart_cropper.py
import numpy as np
from typing import Tuple
from preprocessing.cv_transforms import AdvancedImageProcessor


class SmartCropper:
    """
    Compatibility wrapper for the canonical smart crop/pad implementation.

    The actual laterality-aware crop logic lives in
    `preprocessing.cv_transforms.AdvancedImageProcessor.smart_crop_and_pad()`.
    This class keeps older call sites working while avoiding duplicate logic.

    Attributes:
        target_size (Tuple[int, int]): Desired output resolution (width, height).
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initializes the cropper with target dimensions.
        """
        self.target_size = target_size

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Detects breast tissue, identifies laterality, crops ROI, and pads 
        to target size while maintaining original aspect ratio.

        Args:
            image (np.ndarray): Input grayscale mammogram.

        Returns:
            np.ndarray: Preprocessed and padded image.
        """
        processed_image, _ = AdvancedImageProcessor.smart_crop_and_pad(
            image,
            target_size=self.target_size,
        )
        return processed_image

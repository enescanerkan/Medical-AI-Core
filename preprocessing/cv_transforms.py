# cv_transforms.py
"""
Contains advanced classical computer vision algorithms.
Designed to prepare and enhance medical images before feeding them to deep learning models.
"""

import cv2
import numpy as np
import pywt

class AdvancedImageProcessor:
    """
    A utility class for image preprocessing techniques.
    Adheres to the Single Responsibility Principle (SRP) by only handling image transformations.
    """

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Crucial for enhancing tissue contrast in scans like mammograms or CTs.

        Args:
            image (np.ndarray): Input grayscale image.
            clip_limit (float): Threshold for contrast limiting.
            tile_grid (tuple): Size of grid for histogram equalization.

        Returns:
            np.ndarray: Contrast-enhanced image.
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(image)

    @staticmethod
    def get_discrete_wavelet(image: np.ndarray, wavelet_type: str = 'haar') -> tuple:
        """
        Performs 2D Discrete Wavelet Transform (DWT) to separate image frequencies.
        Useful for isolating calcifications or structural anomalies.

        Args:
            image (np.ndarray): Input grayscale image.
            wavelet_type (str): Type of wavelet to use (e.g., 'haar', 'db1').

        Returns:
            tuple: Approximation (LL) and detail coefficients (LH, HL, HH).
        """
        # Apply DWT
        coeffs2 = pywt.dwt(image, wavelet_type)
        return coeffs2

    @staticmethod
    def find_contours(image: np.ndarray, threshold_val: int = 127) -> list:
        """
        Detects contours in a medical image using basic binary thresholding.
        Serves as a preliminary step for ROI (Region of Interest) extraction or segmentation.

        Args:
            image (np.ndarray): Input grayscale image.
            threshold_val (int): Pixel intensity threshold.

        Returns:
            list: A list of detected contours.
        """

        # Apply binary thresholding
        _, thresh = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)

        # Find contours based on the thresholded image
        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours
# cv_transforms.py
"""
Contains advanced classical computer vision algorithms.
Designed to prepare and enhance medical images before feeding them to deep learning models.
"""

import cv2
import numpy as np
import pywt
from typing import Tuple, List

class AdvancedImageProcessor:
    """
    A utility class for image preprocessing techniques.
    Adheres to the Single Responsibility Principle (SRP) by only handling image transformations.
    """

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Crucial for enhancing tissue contrast in scans like mammograms or CTs.

        Args:
            image (np.ndarray): Input grayscale image.
            clip_limit (float): Threshold for contrast limiting.
            tile_grid (Tuple[int, int]): Size of grid for histogram equalization.

        Returns:
            np.ndarray: Contrast-enhanced image.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(image)

    @staticmethod
    def get_discrete_wavelet(image: np.ndarray, wavelet_type: str = 'haar') -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Performs 2D Discrete Wavelet Transform (DWT) to separate image frequencies.
        Useful for isolating calcifications or structural anomalies.

        Args:
            image (np.ndarray): Input grayscale image.
            wavelet_type (str): Type of wavelet to use (e.g., 'haar', 'db1').

        Returns:
            Tuple: Approximation and detail coefficients.
        """
        coeffs2 = pywt.dwt2(image, wavelet_type)
        return coeffs2

    @staticmethod
    def wavelet_edge_boost(image: np.ndarray, wavelet_type: str = 'haar', boost_factor: float = 2.0) -> np.ndarray:
        """
        Enhances edges and microcalcifications using Inverse Discrete Wavelet Transform (IDWT).
        It decomposes the image, multiplies high-frequency components by a boost factor,
        and reconstructs the image.

        Args:
            image (np.ndarray): Input grayscale image.
            wavelet_type (str): Type of wavelet to use.
            boost_factor (float): Multiplier for high-frequency details (edges).

        Returns:
            np.ndarray: Edge-boosted image.
        """
        # Step 1: Decomposition
        # ll_band: Low freq, lh_band, hl_band, hh_band: High freq (PEP-8 lowercase)
        coeffs2 = pywt.dwt2(image, wavelet_type)
        ll_band, (lh_band, hl_band, hh_band) = coeffs2

        # Step 2: Amplify high-frequency components to boost details
        lh_band = lh_band * boost_factor
        hl_band = hl_band * boost_factor
        hh_band = hh_band * boost_factor

        # Step 3: Reconstruction using IDWT
        reconstructed = pywt.idwt2((ll_band, (lh_band, hl_band, hh_band)), wavelet_type)

        # Step 4: Clip values to valid image range and convert to 8-bit
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        return reconstructed

    @staticmethod
    def find_contours(image: np.ndarray, threshold_val: int = 127) -> List[np.ndarray]:
        """
        Detects contours in a medical image using basic binary thresholding.
        Serves as a preliminary step for ROI (Region of Interest) extraction or segmentation.

        Args:
            image (np.ndarray): Input grayscale image.
            threshold_val (int): Pixel intensity threshold.

        Returns:
            List[np.ndarray]: A list of detected contours.
        """
        _, thresh = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    @staticmethod
    def smart_crop_and_pad(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, str]:
        """
        Detects breast laterality (Left/Right), crops the breast contour to remove
        background noise, and pads it to the target size preserving the aspect ratio.

        Args:
            image (np.ndarray): Input grayscale image.
            target_size (Tuple[int, int]): Desired output dimensions (width, height).

        Returns:
            Tuple[np.ndarray, str]: (Padded Image, Laterality String 'L' or 'R')
        """
        img_h, img_w = image.shape

        # Step 1: Thresholding to isolate the breast tissue (Reusing our function to avoid code duplication)
        contours = AdvancedImageProcessor.find_contours(image, threshold_val=15)

        if not contours:
            return cv2.resize(image, target_size), "Unknown"

        largest_contour = max(contours, key=cv2.contourArea)

        # Step 2: Calculate Center of Mass (Centroid) to determine laterality
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
        else:
            center_x = img_w // 2

        laterality = "L" if center_x < (img_w // 2) else "R"

        # Step 3: Extract bounding box and crop
        x_pos, y_pos, width, height = cv2.boundingRect(largest_contour)
        cropped_breast = image[y_pos:y_pos + height, x_pos:x_pos + width]

        # Step 4: Resize while preserving Aspect Ratio
        scale = min(target_size[0] / width, target_size[1] / height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        resized_breast = cv2.resize(cropped_breast, (new_w, new_h))

        # Step 5: Create a black canvas and apply padding based on laterality
        canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

        # Center vertically
        y_offset = (target_size[1] - new_h) // 2

        # Pad horizontally based on breast position
        if laterality == "L":
            x_offset = 0 # Breast is on the left, pad on the right
        else:
            x_offset = target_size[0] - new_w # Breast is on the right, pad on the left

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_breast

        return canvas, laterality

    @staticmethod
    def calculate_histogram(image: np.ndarray) -> np.ndarray:
        """
        Calculates the 256-bin histogram of a grayscale image.
        This provides a statistical distribution of pixel intensities,
        which is crucial for diagnosing contrast issues before training.

        Args:
            image (np.ndarray): The 8-bit grayscale medical image.

        Returns:
            np.ndarray: A 1D array of length 256 containing pixel counts.
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist.flatten()
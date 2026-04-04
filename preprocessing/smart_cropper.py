# smart_cropper.py
import cv2
import numpy as np
from typing import Tuple


class SmartCropper:
    """
    Advanced preprocessing class for mammography images.
    Implements laterality-aware cropping and aspect ratio preserving padding.

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
        img_h, img_w = image.shape

        # Step 1: Isolate breast tissue using thresholding
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return cv2.resize(image, self.target_size)

        largest_contour = max(contours, key=cv2.contourArea)

        # Step 2: Determine laterality (Left vs Right) using Centroid
        moments = cv2.moments(largest_contour)
        center_x = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else img_w // 2
        laterality = "L" if center_x < (img_w // 2) else "R"

        # Step 3: Crop to bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = image[y:y + h, x:x + w]

        # Step 4: Resize preserving Aspect Ratio
        scale = min(self.target_size[0] / w, self.target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Step 5: Pad onto black canvas based on laterality
        canvas = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        y_offset = (self.target_size[1] - new_h) // 2

        # If Left, align to left (pad right). If Right, align to right (pad left).
        x_offset = 0 if laterality == "L" else self.target_size[0] - new_w
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas